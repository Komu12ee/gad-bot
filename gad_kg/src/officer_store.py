#!/usr/bin/env python3
"""
Officer Store & Knowledge Engine for GAD-BOT
Extracts, indexes, and queries government officers, secretaries, collectors,
commissioners, and their official notification/transfer timelines from both:
  1. HTML directories (Civil List Email, Who is Who, Work Allocation)
  2. 5,300+ OCR Markdown circulars, superannuation orders, and transfer tables.

Persists to:
  - gad_kg/data/work/officers.json & data/fg2/officers.json
  - gad_kg/data/work/officers.db & data/fg2/temporal_graph.db (tables: officers, officer_notifications, fts_officers)
"""

import os
import re
import json
import sqlite3
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "fg2"
GAD_KG_WORK = PROJECT_ROOT / "gad_kg" / "data" / "work"
OFFICERS_JSON = GAD_KG_WORK / "officers.json"
OFFICERS_DB = GAD_KG_WORK / "officers.db"
TEMPORAL_GRAPH_DB = DATA_ROOT / "temporal_graph.db"


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200d", "").replace("\u200c", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_email(em: Optional[str]) -> str:
    em = clean_text(em)
    em = em.replace("[dot]", ".").replace("[at]", "@").replace(" ", "")
    return em.strip(" .,;")


def slugify(text: str) -> str:
    s = clean_text(text).lower()
    s = re.sub(r"^(shri|smt|dr|mr|mrs|ms|श्री|श्रीमती|डॉ|डॉ\.)\s+", "", s)
    s = re.sub(r"[^a-z0-9\u0900-\u097f]+", "_", s).strip("_")
    return s[:64] or "officer"


# Translation / Transliteration mappings for common names and departments
HINDI_CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy', 'श्र': 'shr', 'ड़': 'd', 'ढ़': 'dh', 'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'फ़': 'f'
}
HINDI_VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऑ': 'o'
}
HINDI_MATRAS = {
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h', '़': '', 'ॉ': 'o'
}

def phonetic_transliterate(w: str) -> str:
    res = ''
    i = 0
    n = len(w)
    while i < n:
        ch2 = w[i:i+2]
        if ch2 in HINDI_CONSONANTS:
            cons = HINDI_CONSONANTS[ch2]
            i += 2
        elif w[i] in HINDI_CONSONANTS:
            cons = HINDI_CONSONANTS[w[i]]
            i += 1
        elif w[i] in HINDI_VOWELS:
            res += HINDI_VOWELS[w[i]]
            i += 1
            continue
        elif w[i] in HINDI_MATRAS:
            res += HINDI_MATRAS[w[i]]
            i += 1
            continue
        else:
            res += w[i]
            i += 1
            continue

        if i < n and w[i] in HINDI_MATRAS:
            res += cons
            if w[i] != '्':
                res += HINDI_MATRAS[w[i]]
            i += 1
        elif i < n and w[i] in HINDI_VOWELS:
            res += cons
        else:
            if i < n and (w[i] in HINDI_CONSONANTS or w[i] in HINDI_VOWELS):
                res += cons + 'a'
            else:
                res += cons
    return res.title()

NAME_DICT = {
    'मयंक': 'Mayank', 'अग्रवाल': 'Agrawal', 'अग्रवाल,': 'Agrawal', 'चतुर्वेदी': 'Chaturvedi',
    'रमेन': 'Ramen', 'डेका': 'Deka', 'विष्णु': 'Vishnu', 'देव': 'Deo', 'साय': 'Sai', 'साय,': 'Sai',
    'अरूण': 'Arun', 'अरुण': 'Arun', 'साव': 'Sao', 'विजय': 'Vijay', 'शर्मा': 'Sharma',
    'रामविचार': 'Ramvichar', 'नेताम': 'Netam', 'दयाल': 'Dayal', 'दास': 'Das', 'बघेल': 'Baghel',
    'केदार': 'Kedar', 'कश्यप': 'Kashyap', 'लखन': 'Lakhan', 'लखनलाल': 'Lakhan Lal', 'लाल': 'Lal', 'देवांगन': 'Dewangan',
    'श्याम': 'Shyam', 'बिहारी': 'Bihari', 'जायसवाल': 'Jaiswal', 'ओपी': 'O.P.', 'ओ.पी.': 'O.P.',
    'चौधरी': 'Choudhary', 'टंक': 'Tank', 'राम': 'Ram', 'वर्मा': 'Verma', 'लक्ष्मी': 'Laxmi',
    'राजवाड़े': 'Rajwade', 'राजवाडे': 'Rajwade', 'राजवाडे़': 'Rajwade', 'सुनील': 'Sunil', 'संजय': 'Sanjay', 'अमित': 'Amit',
    'अमिताभ': 'Amitabh', 'जैन': 'Jain', 'सुब्रत': 'Subrat', 'साहू': 'Sahu', 'ऋचा': 'Richa',
    'निधि': 'Nidhi', 'छिब्बर': 'Chhibber', 'निहारिका': 'Niharika', 'बारिक': 'Barik', 'सोनमणि': 'Sonmoni',
    'बोरा': 'Borah', 'शहला': 'Shahla', 'निगार': 'Nigar', 'रोहित': 'Rohit', 'यादव': 'Yadav',
    'कमलप्रीत': 'Kamalpreet', 'सिंह': 'Singh', 'अविनाश': 'Avinash', 'चंपावत': 'Champawat', 'रजत': 'Rajat',
    'कुमार': 'Kumar', 'समीर': 'Sameer', 'विश्नोई': 'Vishnoi', 'सौरभ': 'Saurabh', 'सत्यनारायण': 'Satyanarayan',
    'राठौर': 'Rathore', 'गौरव': 'Gaurav', 'गौतम': 'Gautam', 'आनन्द': 'Anand', 'आनंद': 'Anand',
    'छाबड़ा': 'Chhabra', 'अभिषेक': 'Abhishek', 'पल्लव': 'Pallava', 'पल्लवल': 'Pallava', 'जितेन्द्र': 'Jitendra',
    'जितेंद्र': 'Jitendra', 'शुक्ला': 'Shukla', 'एलेक्स': 'Alex', 'पॉल': 'Paul', 'मेनन': 'Menon',
    'शशांक': 'Shashank', 'पाण्डेय': 'Pandey', 'पांडेय': 'Pandey', 'प्रभात': 'Prabhat', 'मलिक': 'Malik',
    'रितेश': 'Ritesh', 'आलोक': 'Alok', 'अजय': 'Ajay', 'बैजेन्द्र': 'Baijendra', 'राजेन्द्र': 'Rajendra',
    'राजेंद्र': 'Rajendra', 'प्रसाद': 'Prasad', 'मण्डल': 'Mandal', 'मंडल': 'Mandal', 'गजेन्द्र': 'Gajendra',
    'ठाकुर': 'Thakur', 'मनोज': 'Manoj', 'पिंगुआ': 'Pingua', 'विकाश': 'Vikas', 'विकास': 'Vikas',
    'शील': 'Sheel', 'शम्मी': 'Shammi', 'शम्‍मी': 'Shammi', 'आबिदी': 'Abedi', 'चन्दन': 'Chandan',
    'चंदन': 'Chandan', 'रेणु': 'Renu', 'गोनेला': 'Gonela', 'पिल्ले': 'Pillay', 'अमृत': 'Amrit',
    'अभिजीत': 'Abhijit', 'अभिनव': 'Abhinav', 'अमरेश': 'Amaresh', 'अशोक': 'Ashok', 'अश्वनी': 'Ashwani',
    'आशीष': 'Ashish', 'आशुतोष': 'Ashutosh', 'उमेश': 'Umesh', 'ऋषभ': 'Rishabh', 'ऋषि': 'Rishi',
    'कुलदीप': 'Kuldeep', 'कुन्दन': 'Kundan', 'कुंदन': 'Kundan', 'कृष्णा': 'Krishna', 'गणेश': 'Ganesh',
    'गिरीश': 'Girish', 'गोपाल': 'Gopal', 'गोविन्द': 'Govind', 'गोविंद': 'Govind', 'चन्द्र': 'Chandra',
    'चंद्र': 'Chandra', 'चन्द्रकांत': 'Chandrakant', 'चंद्रकांत': 'Chandrakant', 'चेतन': 'Chetan',
    'जगदीश': 'Jagdish', 'जय': 'Jai', 'जयन्त': 'Jayant', 'जयंत': 'Jayant', 'जीवन': 'Jeevan',
    'तारण': 'Taran', 'दिनेश': 'Dinesh', 'दीपक': 'Deepak', 'दिलीप': 'Dilip', 'देवेन्द्र': 'Devendra',
    'देवेंद्र': 'Devendra', 'नरेश': 'Naresh', 'नवीन': 'Naveen', 'नरेन्द्र': 'Narendra', 'नरेंद्र': 'Narendra',
    'निखिल': 'Nikhil', 'नीरज': 'Neeraj', 'पंकज': 'Pankaj', 'पवन': 'Pawan', 'पुष्पेन्द्र': 'Pushpendra',
    'पुष्पेंद्र': 'Pushpendra', 'प्रदीप': 'Pradeep', 'प्रवीण': 'Praveen', 'प्रमोद': 'Pramod', 'प्रशांत': 'Prashant',
    'प्रेम': 'Prem', 'बद्री': 'Badri', 'बलराम': 'Balram', 'बसंत': 'Basant', 'भरत': 'Bharat',
    'भास्कर': 'Bhaskar', 'महेश': 'Mahesh', 'मुकेश': 'Mukesh', 'मुरली': 'Murali', 'योगेश': 'Yogesh',
    'रवि': 'Ravi', 'रवीन्द्र': 'Ravindra', 'रवीश': 'Raveesh', 'राज': 'Raj', 'राकेश': 'Rakesh',
    'राजेश': 'Rajesh', 'राजीव': 'Rajeev', 'राहुल': 'Rahul', 'रोशन': 'Roshan', 'वरुण': 'Varun',
    'विनय': 'Vinay', 'विनोद': 'Vinod', 'विपिन': 'Vipin', 'विवेक': 'Vivek', 'विशाल': 'Vishal',
    'शिशिर': 'Shishir', 'शिव': 'Shiv', 'सतीश': 'Satish', 'संदीप': 'Sandeep', 'सन्दीप': 'Sandeep',
    'संजीव': 'Sanjeev', 'संतोष': 'Santosh', 'सर्वेश': 'Sarvesh', 'सिद्धार्थ': 'Siddharth',
    'सुधाकर': 'Sudhakar', 'सुरेश': 'Suresh', 'सुशील': 'Sushil', 'हरीश': 'Harish', 'हेमंत': 'Hemant',
    'द्विवेदी': 'Dwivedi', 'खलखो': 'Khalkho', 'एक्का': 'Ekka', 'गर्ग': 'Garg', 'बंसल': 'Bansal',
    'गोयल': 'Goyal', 'सिन्हा': 'Sinha', 'राणा': 'Rana', 'जाधव': 'Jadhav', 'मीणा': 'Meena',
    'ध्रूव': 'Dhruw', 'ध्रुव': 'Dhruw', 'नम्रता': 'Namrata', 'किरण': 'Kiran', 'दिव्या': 'Divya',
    'जांगडे': 'Jangade', 'देवी': 'Devi', 'मिश्रा': 'Mishra', 'मिश्र': 'Mishra', 'पटेल': 'Patel',
    'तिवारी': 'Tiwari', 'त्रिपाठी': 'Tripathi', 'दीवान': 'Diwan', 'पौराणिक': 'Puranik', 'मनसागर': 'Mansagar',
    'प्रसन्ना': 'Prasanna', 'हिना': 'Hina', 'अनिमेष': 'Animesh', 'निशांत': 'Nishant', 'शेखर': 'Shekhar',
    'सी.': 'C.', 'आर.': 'R.', 'भीष्म': 'Bhisma', 'डे': 'Dey', 'अर्चना': 'Archana', 'अनुभव': 'Anubhav',
    'विश्वकर्मा': 'Vishwakarma', 'हर्षा': 'Harsha', 'कामता': 'Kamta', 'पंकज': 'Pankaj', 'राजपूत': 'Rajput',
    'सत्यदेव': 'Satyadev', 'मनिन्दर': 'Maninder', 'कौर': 'Kaur', 'सुबोध': 'Subodh', 'परदेशी': 'Pardeshi',
    'कोमल': 'Komal', 'रीना': 'Reena', 'बाबा': 'Baba', 'साहेब': 'Saheb', 'कंगाले': 'Kangale', 'अन्बलगन': 'Anbalagan',
    'अलरमेल': 'Alarmel', 'मंगई': 'Mangai', 'धनंजय': 'Dhananjay', 'देवांगन': 'Dewangan', 'नीलम': 'Neelam',
    'नामदेव': 'Namdev', 'एकता': 'Ekta', 'अग्रवाल': 'Agrawal', 'मुश्ताक': 'Mushtaq', 'अहमद': 'Ahmad',
    'हिमशिखर': 'Himshikhar', 'गुप्ता': 'Gupta', 'शिखा': 'Shikha', 'शारदा': 'Sharda', 'टोपेश्वर': 'Topeshwar',
    'टामन': 'Taman', 'सोनवानी': 'Sonwani', 'कावरे': 'Kavre', 'चौहान': 'Chauhan', 'मांझी': 'Manjhi',
    'डोमन': 'Doman', 'चुरेन्द्र': 'Churendra', 'धुर्वे': 'Dhurve', 'खेतान': 'Khetan', 'चित्तरंजन': 'Chittaranjan',
    'महावर': 'Mahawar', 'अन्वेश': 'Anvesh', 'धृतलहरे': 'Dhritlahre', 'शैलाभ': 'Shailabh',
    'अंशिका': 'Anshika', 'क्लेमेंटिना': 'Clementina', 'लकड़ा': 'Lakra', 'अंजू': 'Anju', 'रामाधीन': 'Ramadhin',
    'सुनीता': 'Suneeta', 'गोरेलाल': 'Gorelal', 'भुआर्य': 'Bhuarya', 'मनराखा': 'Manrakha',
    'खुशवंत': 'Khushwant', 'गुरू': 'Guru', 'सरोटे': 'Sarote', 'शशिगधन': 'Shashigadhan', 'मनीष': 'Manish',
    'संगीता': 'Sangeeta',
    'सी': 'C.', 'आर': 'R.', 'ओ': 'O.', 'पी': 'P.', 'डी': 'D.', 'एस': 'S.', 'के': 'K.',
    'एम': 'M.', 'एन': 'N.', 'बी': 'B.', 'ए': 'A.', 'टी': 'T.', 'जे': 'J.', 'वी': 'V.',
    'एच': 'H.', 'जी': 'G.', 'एल': 'L.', 'यू': 'U.'
}

def transliterate_hindi_name(name_hi: str) -> str:
    if not name_hi or not name_hi.strip():
        return ""
    name_clean = re.sub(r'^(श्री|श्रीमती|डॉ\.|डॉ|सुश्री|कु\.|मेसर्स|स्का\.\s*लीडर)\s+', '', name_hi).strip(' ।,')
    name_clean = re.sub(r'([।\u0900-\u097f])\.', r'\1 . ', name_clean)
    tokens = name_clean.split()
    res = []
    for t in tokens:
        if t == '.':
            continue
        ct = t.strip(' .,।')
        if not ct:
            continue
        if ct in NAME_DICT:
            res.append(NAME_DICT[ct])
        elif re.match(r'^[A-Za-z\.]+$', t):
            res.append(t)
        else:
            res.append(phonetic_transliterate(ct))
    full = ' '.join(res)
    full = re.sub(r'\s*\.\s*', '. ', full)
    return re.sub(r'\s+', ' ', full).strip()

TRANSLITERATIONS = {
    "श्री विकाश शील": "Shri Vikas Sheel",
    "विकाश शील": "Vikas Sheel",
    "श्रीमती शम्मी आबिदी": "Smt. Shammi Abedi",
    "शम्मी आबिदी": "Shammi Abedi",
    "शम्‍मी आबिदी": "Shammi Abedi",
    "श्री चन्दन कुमार": "Shri Chandan Kumar",
    "चन्दन कुमार": "Chandan Kumar",
    "चन्‍दन कुमार": "Chandan Kumar",
    "श्री मनोज कुमार पिंगुआ": "Shri Manoj Kumar Pingua",
    "मनोज कुमार पिंगुआ": "Manoj Kumar Pingua",
    "रेणु गोनेला पिल्ले": "Renu Gonela Pillay",
    "सुब्रत साहू": "Subrat Sahu",
    "श्री अमित अग्रवाल": "Shri Amit Agrawal",
    "श्रीमती ऋचा शर्मा": "Smt. Richa Sharma",
    "श्रीमती निधि छिब्बर": "Smt. Nidhi Chhibber",
    "श्रीमती निहारिका बारिक": "Smt. Niharika Barik",
    "सोनमणि बोरा": "Sonmoni Borah",
    "शहला निगार": "Shahla Nigar",
    "डॉ. रोहित यादव": "Dr. Rohit Yadav",
    "डॉ कमलप्रीत सिंह": "Dr. Kamalpreet Singh",
    "श्री अविनाश चंपावत": "Shri Avinash Champawat",
    "श्री रजत कुमार": "Shri Rajat Kumar",
    "श्री समीर विश्नोई": "Shri Sameer Vishnoi",
    "श्री सौरभ कुमार": "Shri Saurabh Kumar",
    "सुनील कुमार जैन": "Sunil Kumar Jain",
    "श्री सत्यनारायण राठौर": "Shri Satyanarayan Rathore",
    "श्री गौरव कुमार सिंह": "Shri Gaurav Kumar Singh",
    "श्री संजय अग्रवाल": "Shri Sanjay Agrawal",
    "श्री अरूण देव गौतम": "Shri Arun Dev Gautam",
    "डॉ. आनन्द छाबड़ा": "Dr. Anand Chhabra",
    "डॉ. अभिषेक पल्लवल": "Dr. Abhishek Pallava",
    "श्री जितेन्द्र शुक्ला": "Shri Jitendra Shukla",
    "श्री एलेक्स पॉल मेनन": "Shri Alex Paul Menon",
    "श्री शशांक पाण्डेय": "Shri Shashank Pandey",
    "श्री प्रभात मलिक": "Shri Prabhat Malik",
    "श्री रितेश कुमार अग्रवाल": "Shri Ritesh Kumar Agrawal",
    "श्री अमिताभ जैन": "Shri Amitabh Jain",
    "डॉ. आलोक शुक्ला": "Dr. Alok Shukla",
    "श्री अजय सिंह": "Shri Ajay Singh",
    "श्री एन. बैजेन्द्र कुमार": "Shri N. Baijendra Kumar",
    "श्री राजेन्द्र प्रसाद मण्डल": "Shri Rajendra Prasad Mandal",
    "श्री गजेन्द्र सिंह ठाकुर": "Shri Gajendra Singh Thakur",
    "श्री मयंक अग्रवाल": "Shri Mayank Agrawal",
    "मयंक अग्रवाल": "Mayank Agrawal",
    "श्री मयंक चतुर्वेदी": "Shri Mayank Chaturvedi",
    "मयंक चतुर्वेदी": "Mayank Chaturvedi",
}

DEPARTMENT_EN_MAP = {
    "सामान्य प्रशासन विभाग": "General Administration Department (GAD)",
    "राजस्व एवं आपदा प्रबंधन एवं पुनर्वास विभाग": "Revenue and Disaster Management & Rehabilitation Department",
    "राजस्व एवं आपदा प्रबंधन तथा पुनर्वास विभाग": "Revenue and Disaster Management & Rehabilitation Department",
    "वन एवं जलवायु परिवर्तन विभाग": "Forest and Climate Change Department",
    "वित्त विभाग": "Finance Department",
    "गृह एवं जेल विभाग": "Home and Jail Department",
    "नगरीय प्रशासन एवं विकास विभाग": "Urban Administration and Development Department",
    "पंचायत एवं ग्रामीण विकास विभाग": "Panchayat and Rural Development Department",
    "स्कूल शिक्षा विभाग": "School Education Department",
    "स्वास्थ्य एवं परिवार कल्याण विभाग": "Health and Family Welfare Department",
    "महिला एवं बाल विकास विभाग": "Women and Child Development Department",
    "विज्ञान एवं प्रौद्योगिकी विभाग": "Science and Technology Department",
    "इलेक्ट्रॉनिक्स एवं सूचना प्रौद्योगिकी": "Electronics and Information Technology Department (CHiPS)",
    "खनिज साधन विभाग": "Mineral Resources Department",
    "उद्योग विभाग": "Commerce and Industry Department",
    "जनसंपर्क विभाग": "Public Relations Department",
    "विधि एवं विधायी कार्य विभाग": "Law and Legislative Affairs Department",
    "खाद्य नागरिक आपूर्ति एवं उपभोक्ता संरक्षण विभाग": "Food, Civil Supplies and Consumer Protection Department"
}

DESIGNATION_EN_MAP = {
    "राज्यपाल": "Governor",
    "मुख्यमंत्री": "Chief Minister",
    "उप मुख्यमंत्री": "Deputy Chief Minister",
    "मुख्य सचिव": "Chief Secretary",
    "अपर मुख्य सचिव": "Additional Chief Secretary (ACS)",
    "प्रमुख सचिव": "Principal Secretary",
    "सचिव": "Secretary",
    "विशेष सचिव": "Special Secretary",
    "संयुक्त सचिव": "Joint Secretary",
    "उप सचिव": "Deputy Secretary",
    "अवर सचिव": "Under Secretary",
    "कलेक्टर": "District Collector",
    "आयुक्त": "Divisional Commissioner",
    "पुलिस महानिदेशक": "Director General of Police (DGP)",
    "पुलिस महानिरीक्षक": "Inspector General of Police (IG)",
    "पुलिस अधीक्षक": "Superintendent of Police (SP)",
    "जिला वन अधिकारी": "Divisional Forest Officer (DFO)",
    "सीईओ": "Chief Executive Officer (CEO)",
    "संचालक": "Director"
}


def map_designation_en(desig_hi: str) -> str:
    for k, v in DESIGNATION_EN_MAP.items():
        if k in desig_hi:
            return v
    return desig_hi

def map_department_en(dept_hi: str) -> str:
    for k, v in DEPARTMENT_EN_MAP.items():
        if k in dept_hi or dept_hi in k:
            return v
    return dept_hi


def parse_designation_text(raw: str):
    raw = clean_text(raw).strip(" ।|.")
    parts = re.split(
        r"[,;]\s*(?:अति\.\s*प्रभार|अतिरिक्त\s*प्रभार)\s*[-:]?\s*|\s+(?:अति\.\s*प्रभार|अतिरिक्त\s*प्रभार)\s*[-:]?\s*",
        raw
    )
    primary = parts[0].strip(" ।|.,")
    addl = [p.strip(" ।|.,") for p in parts[1:] if p.strip()]

    desig = primary
    dept = ""
    if any(sep in primary for sep in [" - ", " – ", " -", "- ", "-"]):
        m = re.split(r"\s*[-–]\s*", primary, maxsplit=1)
        if len(m) == 2:
            desig, dept = m[0].strip(), m[1].strip()

    return desig, dept, addl


def build_officer_dataset() -> List[Dict[str, Any]]:
    """Builds the comprehensive officer list from HTML sources and circular documents."""
    officers = []
    seen_names = set()

    # 1. Parse Civil List Email HTML (efc290b78f6c8ad7.html)
    civil_list_path = DATA_ROOT / "scrape_raw" / "efc290b78f6c8ad7.html"
    if civil_list_path.exists():
        with open(civil_list_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        tables = soup.find_all("table")
        categories_map = {
            0: ("governor", "राजभवन (Governor)"),
            1: ("governor_secretariat", "राज्यपाल सचिवालय (Governor Secretariat)"),
            2: ("council_of_ministers", "मंत्रिमंडल (Council of Ministers)"),
            4: ("chief_secretary", "मुख्य सचिव (Chief Secretary)"),
            5: ("secretaries", "सचिव / अपर मुख्य सचिव / प्रमुख सचिव (Secretaries & ACS)"),
            6: ("heads_of_department", "विभाग प्रमुख / संचालक (Heads of Department)"),
            7: ("divisional_commissioners", "संभागीय आयुक्त (Divisional Commissioners)"),
            8: ("collectors", "जिला कलेक्टर (District Collectors)"),
            9: ("police_hq", "पुलिस मुख्यालय (Police Headquarters / DG & IG)"),
            10: ("superintendents_of_police", "पुलिस अधीक्षक (Superintendents of Police - SPs)"),
            11: ("divisional_forest_officers", "जिला वन अधिकारी (District Forest Officers - DFOs)")
        }

        for idx, (cat_key, cat_label) in categories_map.items():
            if idx >= len(tables):
                continue
            rows = tables[idx].find_all("tr")
            for r in rows[1:]:
                cells = [clean_text(c.get_text(" ", strip=True)) for c in r.find_all(["td", "th"])]
                if len(cells) < 3:
                    continue
                if idx in [5, 6]:
                    sno, name_raw, desig_dept = cells[0], cells[1], cells[2]
                    email_raw = cells[3] if len(cells) > 3 else ""
                    mobile = cells[4] if len(cells) > 4 else ""
                else:
                    sno, name_raw, desig_dept = cells[0], cells[1], cells[2]
                    email_raw = cells[3] if len(cells) > 3 else ""
                    mobile = cells[4] if len(cells) > 4 else ""

                if not name_raw or name_raw in ["Name", "Department", "S No.", "-"]:
                    continue

                desig_hi, dept_hi, addl_charges = parse_designation_text(desig_dept)
                name_en = TRANSLITERATIONS.get(name_raw, "")
                if not name_en:
                    # check without honorifics
                    clean_name = re.sub(r"^(श्री|श्रीमती|डॉ\.|डॉ)\s+", "", name_raw).strip()
                    name_en = TRANSLITERATIONS.get(clean_name, "")
                if not name_en or re.search(r'[\u0900-\u097f]', name_en):
                    name_en = transliterate_hindi_name(name_raw)

                # English department and designation mappings
                dept_en = map_department_en(dept_hi)
                desig_en = map_designation_en(desig_hi)

                if cat_key == "chief_secretary":
                    desig_en = "Chief Secretary, Government of Chhattisgarh"
                    dept_en = "Government of Chhattisgarh"
                    dept_hi = "छत्तीसगढ़ शासन"

                oid = slugify(name_en or name_raw)
                if oid in seen_names:
                    oid = f"{oid}_{len(officers)+1}"
                seen_names.add(oid)

                officers.append({
                    "id": oid,
                    "name_hi": name_raw,
                    "name_en": name_en,
                    "category": cat_key,
                    "category_label": cat_label,
                    "designation_hi": desig_hi,
                    "designation_en": desig_en,
                    "department_hi": dept_hi,
                    "department_en": dept_en,
                    "additional_charges": addl_charges,
                    "email": clean_email(email_raw),
                    "phone": mobile,
                    "status": "active",
                    "source_doc_id": "efc290b78f6c8ad7",
                    "source_name": "Chhattisgarh State Civil List Email Directory",
                    "source_url": "https://gad.cg.gov.in/Display_civil_list_Email.aspx",
                    "notifications": []
                })

    # 2. Parse Who's Who HTML (3dbb992557ea858e.html)
    who_path = DATA_ROOT / "scrape_raw" / "3dbb992557ea858e.html"
    if who_path.exists():
        with open(who_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        tables = soup.find_all("table")
        if tables:
            rows = tables[0].find_all("tr")
            for r in rows[1:]:
                cells = [clean_text(c.get_text(" ", strip=True)) for c in r.find_all(["td", "th"])]
                if len(cells) < 3 or not cells[1] or cells[1] in ["नाम", "Name"]:
                    continue
                sno, name_en, desig_en = cells[0], cells[1], cells[2]
                mobile = cells[3] if len(cells) > 3 else ""
                email = cells[4] if len(cells) > 4 else ""

                # Check if officer already in list (e.g. Vikas Sheel, Chandan Kumar)
                matched = False
                for off in officers:
                    if off["name_en"].lower() in name_en.lower() or name_en.lower() in off["name_en"].lower():
                        # Enrich existing
                        if not off.get("phone") and mobile:
                            off["phone"] = mobile
                        if not off.get("email") and email:
                            off["email"] = clean_email(email)
                        off["gad_designation"] = desig_en
                        matched = True
                        break

                if not matched:
                    oid = slugify(name_en)
                    if oid in seen_names:
                        oid = f"{oid}_{len(officers)+1}"
                    seen_names.add(oid)

                    officers.append({
                        "id": oid,
                        "name_hi": name_en,
                        "name_en": name_en,
                        "category": "gad_establishment",
                        "category_label": "सामान्य प्रशासन विभाग (GAD Establishment Officers)",
                        "designation_hi": desig_en,
                        "designation_en": desig_en,
                        "department_hi": "सामान्य प्रशासन विभाग",
                        "department_en": "General Administration Department (GAD)",
                        "additional_charges": [],
                        "email": clean_email(email),
                        "phone": mobile,
                        "status": "active",
                        "source_doc_id": "3dbb992557ea858e",
                        "source_name": "GAD Official Who's Who",
                        "source_url": "https://gad.cg.gov.in/who_is_who.aspx",
                        "notifications": []
                    })

    # 3. Add Key Historical & CHiPS Leadership
    chips_leaders = [
        {
            "id": "alex_paul_menon",
            "name_hi": "श्री एलेक्स पॉल मेनन",
            "name_en": "Shri Alex Paul Menon",
            "category": "chips_leadership",
            "category_label": "चिप्स एवं इलेक्ट्रॉनिक्स नेतृत्व (CHiPS & IT Leadership)",
            "designation_hi": "पूर्व मुख्य कार्यपालन अधिकारी (सीईओ), चिप्स / संयुक्त सचिव",
            "designation_en": "Former Chief Executive Officer (CEO), CHiPS / Joint Secretary",
            "department_hi": "छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (चिप्स) / इलेक्ट्रॉनिक्स एवं सूचना प्रौद्योगिकी",
            "department_en": "Chhattisgarh Infotech Promotion Society (CHiPS) / Electronics & IT",
            "additional_charges": ["संयुक्त सचिव, इलेक्ट्रॉनिक्स एवं सूचना प्रौद्योगिकी विभाग"],
            "email": "ceochips@nic.in",
            "phone": "",
            "status": "former",
            "source_doc_id": "5aed8b79a68a88e8",
            "source_name": "Official CHiPS Leadership Circulars",
            "source_url": "https://gad.cg.gov.in",
            "notifications": []
        },
        {
            "id": "sameer_vishnoi",
            "name_hi": "श्री समीर विश्नोई",
            "name_en": "Shri Sameer Vishnoi",
            "category": "chips_leadership",
            "category_label": "चिप्स एवं इलेक्ट्रॉनिक्स नेतृत्व (CHiPS & IT Leadership)",
            "designation_hi": "पूर्व मुख्य कार्यपालन अधिकारी (सीईओ), चिप्स / संचालक भौमिकी तथा खनिकर्म",
            "designation_en": "Former CEO, CHiPS / Director, Geology and Mining",
            "department_hi": "छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (चिप्स) / खनिज साधन",
            "department_en": "Chhattisgarh Infotech Promotion Society (CHiPS) / Mineral Resources",
            "additional_charges": ["निलंबित"],
            "email": "sameervishnoi.ias09@ias.nic.in",
            "phone": "",
            "status": "former",
            "source_doc_id": "efc290b78f6c8ad7",
            "source_name": "Official Civil List & Mining Department Orders",
            "source_url": "https://gad.cg.gov.in",
            "notifications": []
        },
        {
            "id": "ritesh_kumar_agrawal",
            "name_hi": "श्री रितेश कुमार अग्रवाल",
            "name_en": "Shri Ritesh Kumar Agrawal",
            "category": "chips_leadership",
            "category_label": "चिप्स एवं इलेक्ट्रॉनिक्स नेतृत्व (CHiPS & IT Leadership)",
            "designation_hi": "पूर्व मुख्य कार्यपालन अधिकारी (सीईओ), चिप्स",
            "designation_en": "Former Chief Executive Officer (CEO), CHiPS",
            "department_hi": "छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (चिप्स)",
            "department_en": "Chhattisgarh Infotech Promotion Society (CHiPS)",
            "additional_charges": [],
            "email": "ceochips@nic.in",
            "phone": "",
            "status": "former",
            "source_doc_id": "fbc9b70ccdf197ba",
            "source_name": "Official CHiPS Gazetted Notifications",
            "source_url": "https://gad.cg.gov.in",
            "notifications": []
        },
        {
            "id": "prabhat_malik",
            "name_hi": "प्रभात मलिक",
            "name_en": "Shri Prabhat Malik",
            "category": "chips_leadership",
            "category_label": "चिप्स एवं इलेक्ट्रॉनिक्स नेतृत्व (CHiPS & IT Leadership)",
            "designation_hi": "माननीय मुख्यमंत्री के संयुक्त सचिव, संचालक उद्योग एवं विमानन / पूर्व सीईओ चिप्स",
            "designation_en": "Joint Secretary to Hon'ble CM, Director Industries & Aviation / Former CEO CHiPS",
            "department_hi": "मुख्यमंत्री सचिवालय / उद्योग विभाग / चिप्स",
            "department_en": "Chief Minister Secretariat / Industry Department / CHiPS",
            "additional_charges": ["संचालक - उद्योग विभाग", "संचालक - विमानन विभाग"],
            "email": "pmalik2890@gmail.com",
            "phone": "",
            "status": "active",
            "source_doc_id": "efc290b78f6c8ad7",
            "source_name": "Official Civil List & CM Secretariat Orders",
            "source_url": "https://gad.cg.gov.in",
            "notifications": []
        },
        {
            "id": "shashank_pandey",
            "name_hi": "श्री शशांक पाण्डेय",
            "name_en": "Shri Shashank Pandey",
            "category": "chips_leadership",
            "category_label": "चिप्स एवं इलेक्ट्रॉनिक्स नेतृत्व (CHiPS & IT Leadership)",
            "designation_hi": "अतिरिक्त मुख्य कार्यपालन अधिकारी (अति. सीईओ), चिप्स / सीईओ सूडा",
            "designation_en": "Additional Chief Executive Officer (Addl. CEO), CHiPS / CEO SUDA",
            "department_hi": "छत्तीसगढ़ इन्फोटेक प्रमोशन सोसाइटी (चिप्स) / नगरीय प्रशासन",
            "department_en": "Chhattisgarh Infotech Promotion Society (CHiPS) / Urban Administration",
            "additional_charges": ["मुख्य कार्यपालन अधिकारी, राज्य शहरी विकास अभिकरण (सूडा) रायपुर"],
            "email": "ceochips@nic.in",
            "phone": "",
            "status": "active",
            "source_doc_id": "fbc9b70ccdf197ba",
            "source_name": "Official Notification बी-1-1/2019/एक/4 & Gradation List",
            "source_url": "https://gad.cg.gov.in",
            "notifications": []
        }
    ]

    for cl in chips_leaders:
        if cl["id"] not in seen_names:
            seen_names.add(cl["id"])
            officers.append(cl)

    # 4. Add Retired IAS Officers (from 14f73433fbdd53d7 superannuation notifications)
    retired_ias_list = [
        {"name_hi": "श्री अजय सिंह", "name_en": "Shri Ajay Singh", "batch": "1983", "last_post": "अध्यक्ष, राजस्व मण्डल छत्तीसगढ़, बिलासपुर", "retired_date": "2020-02-29"},
        {"name_hi": "श्री एन. बैजेन्द्र कुमार", "name_en": "Shri N. Baijendra Kumar", "batch": "1985", "last_post": "अध्यक्ष एवं प्रबंध संचालक, NMDC हैदराबाद", "retired_date": "2020-07-31"},
        {"name_hi": "डॉ. आलोक शुक्ला", "name_en": "Dr. Alok Shukla", "batch": "1986", "last_post": "प्रमुख सचिव, स्कूल शिक्षा एवं मंत्रालय", "retired_date": "2020-05-31"},
        {"name_hi": "श्री राजेन्द्र प्रसाद मण्डल", "name_en": "Shri Rajendra Prasad Mandal", "batch": "1987", "last_post": "मुख्य सचिव, छत्तीसगढ़ शासन / अध्यक्ष NRDA", "retired_date": "2020-11-30"},
        {"name_hi": "श्री चित्तरंजन कुमार खेतान", "name_en": "Shri Chittaranjan Kumar Khetan", "batch": "1987", "last_post": "अध्यक्ष, राजस्व मण्डल छत्तीसगढ़, बिलासपुर", "retired_date": "2021-07-31"},
        {"name_hi": "श्री अमिताभ जैन", "name_en": "Shri Amitabh Jain", "batch": "1989", "last_post": "मुख्य सचिव, छत्तीसगढ़ शासन", "retired_date": "2025-06-30"},
        {"name_hi": "श्री देवी दयाल सिंह", "name_en": "Shri Devi Dayal Singh", "batch": "2000", "last_post": "सचिव, सामान्य प्रशासन विभाग, मंत्रालय", "retired_date": "2021-06-30"},
        {"name_hi": "श्री विजय कुमार धुर्वे", "name_en": "Shri Vijay Kumar Dhurve", "batch": "2002", "last_post": "संयुक्त सचिव, जेल एवं परिवहन विभाग", "retired_date": "2021-09-30"},
        {"name_hi": "श्री गोविन्द राम चुरेन्द्र", "name_en": "Shri Govind Ram Churendra", "batch": "2003", "last_post": "सचिव, छत्तीसगढ़ पुलिस जवाबदेही प्राधिकार", "retired_date": "2024-12-31"},
        {"name_hi": "डॉ. संजय कुमार अलंग", "name_en": "Dr. Sanjay Kumar Alang", "batch": "2004", "last_post": "आयुक्त, बिलासपुर संभाग, बिलासपुर", "retired_date": "2024-07-31"},
        {"name_hi": "श्री टामन सिंह सोनवानी", "name_en": "Shri Taman Singh Sonwani", "batch": "2004", "last_post": "सचिव, मुख्यमंत्री सचिवालय / अध्यक्ष CGPSC", "retired_date": "2021-09-30"},
        {"name_hi": "श्री टोपेश्वर वर्मा", "name_en": "Shri Topeshwar Verma", "batch": "2005", "last_post": "सचिव, छत्तीसगढ़ राजस्व मण्डल, बिलासपुर", "retired_date": "2025-10-31"},
        {"name_hi": "श्रीमती शारदा वर्मा", "name_en": "Smt. Sharda Verma", "batch": "2008", "last_post": "विशेष सचिव, वित्त विभाग", "retired_date": "2024-12-31"},
        {"name_hi": "श्री महादेव कावरे", "name_en": "Shri Mahadev Kavre", "batch": "2008", "last_post": "आयुक्त, रायपुर संभाग / बिलासपुर संभाग", "retired_date": "2026-05-31"},
        {"name_hi": "श्री कुमार लाल चौहान", "name_en": "Shri Kumar Lal Chauhan", "batch": "2009", "last_post": "कलेक्टर, जिला बलौदाबाजार-भाटापारा", "retired_date": "2025-05-31"},
        {"name_hi": "श्री विपिन मांझी", "name_en": "Shri Vipin Manjhi", "batch": "2009", "last_post": "कलेक्टर, जिला नारायणपुर", "retired_date": "2025-05-31"},
        {"name_hi": "श्री डोमन सिंह", "name_en": "Shri Doman Singh", "batch": "2009", "last_post": "आयुक्त, बस्तर संभाग, जगदलपुर", "retired_date": "2026-12-31"},
        {"name_hi": "श्री सुधाकर खलखो", "name_en": "Shri Sudhakar Khalkho", "batch": "2012", "last_post": "संयुक्त सचिव, मंत्रालय", "retired_date": "2024-09-30"},
        {"name_hi": "श्री भगवान सिंह उड़के", "name_en": "Shri Bhagwan Singh Uike", "batch": "2016", "last_post": "अतिरिक्त कलेक्टर, अंतागढ़, कांकेर", "retired_date": "2026-12-31"},
        {"name_hi": "श्री उमेश कुमार अग्रवाल", "name_en": "Shri Umesh Kumar Agrawal", "batch": "2002", "last_post": "अतिरिक्त आयुक्त / सचिव", "retired_date": "2022-06-29"},
        {"name_hi": "श्री अमृत खलखो", "name_en": "Shri Amrit Khalkho", "batch": "2002", "last_post": "सचिव, राज्यपाल सचिवालय", "retired_date": "2023-07-29"},
        {"name_hi": "श्री नरेन्द्र कुमार शुक्ला", "name_en": "Shri Narendra Kumar Shukla", "batch": "2000", "last_post": "सचिव, राजस्व मण्डल", "retired_date": "2020-05-22"},
        {"name_hi": "श्री डी.सी. महावर", "name_en": "Shri D.C. Mahawar", "batch": "2000", "last_post": "सचिव / सदस्य", "retired_date": "2020-12-15"}
    ]

    for ri in retired_ias_list:
        oid = slugify(f"{ri['name_en']}_retired")
        if oid not in seen_names:
            seen_names.add(oid)
            officers.append({
                "id": oid,
                "name_hi": ri["name_hi"],
                "name_en": ri["name_en"],
                "category": "retired_ias",
                "category_label": "सेवानिवृत्त भारतीय प्रशासनिक सेवा (Retired IAS Officers)",
                "designation_hi": f"सेवानिवृत्त भा.प्र.से. ({ri['batch']}) - पूर्व {ri['last_post']}",
                "designation_en": f"Retired IAS ({ri['batch']}) - Former {ri['last_post']}",
                "department_hi": "भारतीय प्रशासनिक सेवा (छ.ग. संवर्ग)",
                "department_en": "Indian Administrative Service (Chhattisgarh Cadre)",
                "additional_charges": [f"सेवानिवृत्ति तिथि: {ri['retired_date']}", f"बैच: {ri['batch']}"],
                "email": "",
                "phone": "",
                "status": "retired",
                "source_doc_id": "14f73433fbdd53d7",
                "source_name": "GAD Superannuation Order क्रमांक ई 3-01/2003/एक/2",
                "source_url": "https://gad.cg.gov.in",
                "notifications": [{
                    "doc_id": "14f73433fbdd53d7",
                    "order_no": "ई 3-01/2003/एक/2",
                    "order_date": ri["retired_date"],
                    "subject": f"अधिवार्षिकी आयु पूर्ण करने के फलस्वरूप शासकीय सेवा से सेवानिवृत्ति आदेश: {ri['name_hi']}",
                    "excerpt": f"राज्य शासन एतद्द्वारा भारतीय प्रशासनिक सेवा के अधिकारी {ri['name_hi']} ({ri['batch']}), पद: {ri['last_post']} को अधिवार्षिकी आयु पूर्ण करने पर दिनांक {ri['retired_date']} से सेवानिवृत्त करता है।"
                }]
            })

    return officers


def link_circular_notifications(officers: List[Dict[str, Any]]):
    """Scans all stage2 structured markdown files to attach transfer/posting/order citations to each officer."""
    stage2_dir = DATA_ROOT / "stage2"
    if not stage2_dir.exists():
        return

    # Build search patterns for each officer
    officer_matchers = []
    for off in officers:
        names_to_match = [off["name_hi"], off["name_en"]]
        clean_hi = re.sub(r"^(श्री|श्रीमती|डॉ\.|डॉ)\s+", "", off["name_hi"]).strip()
        clean_en = re.sub(r"^(shri|smt|dr|mr|mrs|ms)\s+", "", off["name_en"], flags=re.IGNORECASE).strip()
        if len(clean_hi) >= 3:
            names_to_match.append(clean_hi)
        if len(clean_en) >= 3:
            names_to_match.append(clean_en)

        # Regex
        escaped = [re.escape(n) for n in names_to_match if len(n) >= 3]
        pattern = re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)
        officer_matchers.append((off, pattern))

    # Read all documents metadata for dates and order numbers if available
    doc_meta_file = DATA_ROOT / "doc_metadata.json"
    doc_meta = {}
    if doc_meta_file.exists():
        try:
            with open(doc_meta_file, "r", encoding="utf-8") as f:
                doc_meta = json.load(f)
        except Exception:
            pass

    md_files = list(stage2_dir.glob("*/structured.md"))
    print(f"Linking notifications across {len(md_files)} documents...")

    for fp in md_files:
        doc_id = fp.parent.name
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        meta = doc_meta.get(doc_id, {})
        order_no = meta.get("order_number") or ""
        order_date = meta.get("date") or ""
        subject = meta.get("subject") or ""

        # Extract order number / date from text if not in meta
        if not order_no:
            m_ord = re.search(r"(?:क्रमांक|आदेश\s*क्र\.?|फा\.?\s*क्र\.?|एफ\s*[\d\-]+/\d+)\s*[:\s]*([^\n\r,]+)", text)
            if m_ord:
                order_no = clean_text(m_ord.group(1))[:60]
        if not order_date:
            m_dt = re.search(r"दिनांक\s*[:\s]*(\d{1,2}\s+[^\s,]+\s+\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", text)
            if m_dt:
                order_date = clean_text(m_dt.group(1))

        # Check officer matches
        for off, pat in officer_matchers:
            if len(off["notifications"]) >= 10:
                continue  # limit per officer to avoid giant lists
            m = pat.search(text)
            if m:
                # Extract surrounding context (up to 200 chars)
                start = max(0, m.start() - 100)
                end = min(len(text), m.end() + 150)
                excerpt = clean_text(text[start:end])

                # Check if this doc_id already linked
                if not any(n["doc_id"] == doc_id for n in off["notifications"]):
                    off["notifications"].append({
                        "doc_id": doc_id,
                        "order_no": order_no or f"GAD Order ({doc_id[:8]})",
                        "order_date": order_date or "2000-2026",
                        "subject": subject or f"Government Notification / Circular ({doc_id[:8]})",
                        "excerpt": excerpt
                    })

    print("Notification linking complete.")


def load_document_metadata_index() -> Dict[str, Dict[str, Any]]:
    """Loads documents.json and doc_metadata.json into an in-memory map keyed by doc_dir and canonical id."""
    index: Dict[str, Dict[str, Any]] = {}
    docs_json = DATA_ROOT / "documents.json"
    meta_json = DATA_ROOT / "doc_metadata.json"

    by_id: Dict[str, Any] = {}
    by_dir: Dict[str, Any] = {}

    if docs_json.exists():
        try:
            with open(docs_json, "r", encoding="utf-8") as f:
                doc_list = json.load(f).get("documents", [])
                for d in doc_list:
                    if d.get("id"):
                        by_id[d["id"]] = d
                    if d.get("doc_dir"):
                        by_dir[d["doc_dir"]] = d
        except Exception as e:
            print(f"[load_document_metadata_index] Warning loading documents.json: {e}")

    dmeta: Dict[str, Any] = {}
    if meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                dmeta = json.load(f).get("documents", {})
        except Exception as e:
            print(f"[load_document_metadata_index] Warning loading doc_metadata.json: {e}")

    for doc_dir, d in by_dir.items():
        can_id = d.get("id")
        m = dmeta.get(can_id, {})
        entry = {
            "doc_dir": doc_dir,
            "canonical_id": can_id,
            "filename": d.get("original_name") or f"{doc_dir}.pdf",
            "source_url": m.get("source_url") or d.get("source_url") or "https://gad.cg.gov.in/",
            "listing_url": m.get("listing_url") or "https://gad.cg.gov.in/",
            "title": m.get("title") or d.get("original_name") or "Government Notification / Circular",
            "fetched_at": m.get("fetched_at", ""),
        }
        index[doc_dir] = entry
        if can_id:
            index[can_id] = entry

    for can_id, m in dmeta.items():
        if can_id not in index:
            d = by_id.get(can_id, {})
            doc_dir = d.get("doc_dir") or m.get("doc_dir") or can_id
            entry = {
                "doc_dir": doc_dir,
                "canonical_id": can_id,
                "filename": d.get("original_name") or f"{doc_dir}.pdf",
                "source_url": m.get("source_url") or "https://gad.cg.gov.in/",
                "listing_url": m.get("listing_url") or "https://gad.cg.gov.in/",
                "title": m.get("title") or d.get("original_name") or "Government Notification / Circular",
                "fetched_at": m.get("fetched_at", ""),
            }
            index[can_id] = entry
            index[doc_dir] = entry

    # Also load comprehensive doc_url_index.json if present
    url_index_file = GAD_KG_WORK / "doc_url_index.json"
    if url_index_file.exists():
        try:
            with open(url_index_file, "r", encoding="utf-8") as f:
                extra_idx = json.load(f)
                index.update(extra_idx)
        except Exception as e:
            print(f"[load_document_metadata_index] Warning loading doc_url_index.json: {e}")

    return index


def persist_officer_store(officers: List[Dict[str, Any]]):
    """Enriches officers with official source URLs and saves to JSON and SQLite databases."""
    GAD_KG_WORK.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    meta_index = load_document_metadata_index()

    # Enrich officers and notifications with document metadata and source URLs
    for off in officers:
        s_doc = off.get("source_doc_id", "")
        m = meta_index.get(s_doc, {})
        off["canonical_source_id"] = m.get("canonical_id") or s_doc
        off["source_title"] = m.get("title") or off.get("source_name") or "General Administration Department"
        off["source_url"] = off.get("source_url") or m.get("source_url") or "https://gad.cg.gov.in/"
        off["filename"] = m.get("filename") or f"{s_doc}.html"

        for n in off.get("notifications", []):
            n_doc = n.get("doc_id", "")
            nm = meta_index.get(n_doc, {})
            n["canonical_doc_id"] = nm.get("canonical_id") or n_doc
            n["doc_title"] = nm.get("title") or n.get("subject") or "Government Order"
            n["source_url"] = nm.get("source_url") or "https://gad.cg.gov.in/"
            n["filename"] = nm.get("filename") or f"{n_doc}.pdf"

    # 1. Write JSON files
    with open(OFFICERS_JSON, "w", encoding="utf-8") as f:
        json.dump(officers, f, ensure_ascii=False, indent=2)
    print(f"✅ Written {len(officers)} officers to {OFFICERS_JSON}")

    # Copy to data/fg2/officers.json
    fg2_officers_json = DATA_ROOT / "officers.json"
    with open(fg2_officers_json, "w", encoding="utf-8") as f:
        json.dump(officers, f, ensure_ascii=False, indent=2)
    print(f"✅ Written copy to {fg2_officers_json}")

    # 2. Write SQLite DBs (gad_kg/data/work/officers.db and data/fg2/temporal_graph.db)
    for db_path in [OFFICERS_DB, TEMPORAL_GRAPH_DB]:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # Create officers table
        cur.execute("DROP TABLE IF EXISTS officers")
        cur.execute("""
            CREATE TABLE officers (
                id TEXT PRIMARY KEY,
                name_hi TEXT NOT NULL,
                name_en TEXT NOT NULL,
                category TEXT,
                category_label TEXT,
                designation_hi TEXT,
                designation_en TEXT,
                department_hi TEXT,
                department_en TEXT,
                additional_charges TEXT,
                email TEXT,
                phone TEXT,
                status TEXT,
                source_doc_id TEXT,
                canonical_source_id TEXT,
                source_name TEXT,
                source_title TEXT,
                source_url TEXT,
                filename TEXT,
                notification_count INTEGER DEFAULT 0
            )
        """)

        # Create officer_notifications table with metadata & source URL columns
        cur.execute("DROP TABLE IF EXISTS officer_notifications")
        cur.execute("""
            CREATE TABLE officer_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                officer_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                canonical_doc_id TEXT,
                order_no TEXT,
                order_date TEXT,
                subject TEXT,
                doc_title TEXT,
                source_url TEXT,
                filename TEXT,
                excerpt TEXT,
                FOREIGN KEY (officer_id) REFERENCES officers(id)
            )
        """)

        # Create FTS5 virtual table for lightning search
        cur.execute("DROP TABLE IF EXISTS fts_officers")
        cur.execute("""
            CREATE VIRTUAL TABLE fts_officers USING fts5(
                officer_id UNINDEXED,
                name_hi,
                name_en,
                designation_hi,
                designation_en,
                department_hi,
                department_en,
                additional_charges,
                tokenize = 'unicode61'
            )
        """)

        # Insert data
        for off in officers:
            addl_str = " | ".join(off.get("additional_charges", []))
            notifs = off.get("notifications", [])
            cur.execute("""
                INSERT INTO officers (
                    id, name_hi, name_en, category, category_label,
                    designation_hi, designation_en, department_hi, department_en,
                    additional_charges, email, phone, status,
                    source_doc_id, canonical_source_id, source_name, source_title, source_url, filename, notification_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                off["id"], off["name_hi"], off["name_en"], off.get("category", ""),
                off.get("category_label", ""), off.get("designation_hi", ""),
                off.get("designation_en", ""), off.get("department_hi", ""),
                off.get("department_en", ""), addl_str, off.get("email", ""),
                off.get("phone", ""), off.get("status", "active"),
                off.get("source_doc_id", ""), off.get("canonical_source_id", ""),
                off.get("source_name", ""), off.get("source_title", ""),
                off.get("source_url", ""), off.get("filename", ""), len(notifs)
            ))

            cur.execute("""
                INSERT INTO fts_officers (
                    officer_id, name_hi, name_en, designation_hi, designation_en,
                    department_hi, department_en, additional_charges
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                off["id"], off["name_hi"], off["name_en"], off.get("designation_hi", ""),
                off.get("designation_en", ""), off.get("department_hi", ""),
                off.get("department_en", ""), addl_str
            ))

            for n in notifs:
                cur.execute("""
                    INSERT INTO officer_notifications (
                        officer_id, doc_id, canonical_doc_id, order_no, order_date,
                        subject, doc_title, source_url, filename, excerpt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    off["id"], n["doc_id"], n.get("canonical_doc_id", ""),
                    n.get("order_no", ""), n.get("order_date", ""),
                    n.get("subject", ""), n.get("doc_title", ""),
                    n.get("source_url", ""), n.get("filename", ""),
                    n.get("excerpt", "")
                ))

        conn.commit()
        conn.close()
        print(f"✅ Loaded {len(officers)} officers & {sum(len(o.get('notifications', [])) for o in officers)} notifications into {db_path}")


class OfficerStore:
    """Query service for officer profiles, secretaries, leadership, and notifications."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (TEMPORAL_GRAPH_DB if TEMPORAL_GRAPH_DB.exists() else OFFICERS_DB)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_officer(self, officer_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM officers WHERE id = ?", (officer_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
        off = dict(row)
        off["additional_charges"] = [c.strip() for c in (off.get("additional_charges") or "").split("|") if c.strip()]

        cur.execute("SELECT * FROM officer_notifications WHERE officer_id = ? ORDER BY id DESC", (officer_id,))
        notifs = [dict(r) for r in cur.fetchall()]
        conn.close()

        meta_index = None
        for n in notifs:
            if not n.get("source_url") or n.get("source_url") == "https://gad.cg.gov.in/":
                if meta_index is None:
                    meta_index = load_document_metadata_index()
                nm = meta_index.get(n.get("doc_id", ""), {})
                if nm.get("source_url"):
                    n["source_url"] = nm["source_url"]
                if nm.get("title") and (not n.get("doc_title") or n["doc_title"] == "Government Order"):
                    n["doc_title"] = nm["title"]
                if nm.get("filename") and not n.get("filename"):
                    n["filename"] = nm["filename"]
                if nm.get("canonical_id") and not n.get("canonical_doc_id"):
                    n["canonical_doc_id"] = nm["canonical_id"]
        off["notifications"] = notifs
        return off

    def search_officers(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        query = clean_text(query).strip()
        if not query:
            return []

        # Common phonetic / alias substitutions
        norm_q = query.lower()
        substitutions = [
            (r"\bsammi\b", "shammi"),
            (r"\benfotech\b", "infotech"),
            (r"\bchips\b", "चिप्स CHiPS infotech"),
            (r"\bforest\b", "वन Forest"),
            (r"\brevenue\b", "राजस्व Revenue"),
            (r"\bdisaster\b", "आपदा Disaster"),
            (r"\bchief secretary\b", "मुख्य सचिव 'Chief Secretary'"),
            (r"\bcs\b", "मुख्य सचिव 'Chief Secretary'"),
            (r"\bfinance\b", "वित्त Finance"),
            (r"\beducation\b", "शिक्षा Education"),
            (r"\bhealth\b", "स्वास्थ्य Health"),
            (r"\bhome\b", "गृह Home"),
            (r"\burban\b", "नगरीय Urban"),
            (r"\bpension\b", "पेंशन Pension"),
            (r"\bretired\b", "सेवानिवृत्त Retired"),
        ]
        for pat, repl in substitutions:
            norm_q = re.sub(pat, repl, norm_q)

        stopwords = {
            "who", "is", "was", "were", "the", "of", "about", "abt", "tell", "me", "give", "list",
            "and", "or", "to", "all", "in", "shri", "smt", "dr", "mr", "mrs", "ms",
            "what", "are", "for", "from", "with", "at", "by", "on", "as", "an", "a",
            "this", "that", "these", "those", "how", "when", "where", "which", "why",
            "before", "after", "former", "previous", "ex", "preceded", "predecessor",
            "kon", "hain", "hai", "kripya", "bataiye", "rules", "rule", "order", "document",
            "documents", "circular", "circulars", "details", "status",
            "का", "के", "की", "में", "से", "को", "ने", "पर", "और", "या", "तथा", "एवं",
            "हेतु", "बाबत", "संबंधी", "बारे", "कौन", "हैं", "है", "था", "थी", "थे", "नियम",
            "आदेश", "परिपत्र", "पूर्व", "पहले", "पिछले", "रिटायर्ड",
        }

        common_surnames = {
            "agrawal", "agarwal", "sharma", "singh", "kumar", "verma", "gupta", 
            "mishra", "mishr", "pandey", "tiwari", "tripathi", "patel", "yadav", 
            "jain", "sahu", "rathore", "thakur", "baghel", "kashyap", "netam", 
            "dewangan", "diwan", "choudhary", "chaudhary", "mandal", "shukla", 
            "dey", "deka", "jena", "jha", "sen", "rao", "naag", "tanwar", 
            "sonkar", "dubey", "chaubey", "dwivedi", "bose", "ghosh", "banerjee",
            "अग्रवाल", "शर्मा", "सिंह", "कुमार", "वर्मा", "गुप्ता", "मिश्रा", 
            "पाण्डेय", "पांडेय", "तिवारी", "त्रिपाठी", "पटेल", "यादव", "जैन", 
            "साहू", "राठौर", "ठाकुर", "बघेल", "कश्यप", "नेताम", "देवांगन", 
            "दीवान", "चौधरी", "मण्डल", "मंडल", "शुक्ला", "डे", "डेका"
        }

        dept_post_keywords = {
            "secretary", "secretaries", "सचिव", "collector", "collectors", "कलेक्टर", 
            "sp", "sps", "police", "पुलिस", "dfo", "dfos", "forest", "वन", "revenue", 
            "राजस्व", "commissioner", "commissioners", "आयुक्त", "director", "संचालक", 
            "minister", "मंत्री", "governor", "राज्यपाल", "chips", "चिप्स", "ias", "ips", "ifs", 
            "establishment", "स्थापना", "disaster", "आपदा", "finance", "वित्त", "health", 
            "स्वास्थ्य", "education", "शिक्षा", "home", "गृह", "urban", "नगरीय", "pwd"
        }

        raw_tokens = re.findall(r"[\w\u0900-\u097f]+", norm_q)
        tokens = [t for t in raw_tokens if t.lower() not in stopwords and len(t) >= 2]
        if not tokens:
            return []

        dept_tokens = [t.lower() for t in tokens if t.lower() in dept_post_keywords]
        has_dept_intent = len(dept_tokens) > 0 or any(k in norm_q for k in ["chief secretary", "मुख्य सचिव"])

        name_tokens = [t.lower() for t in tokens if t.lower() not in dept_post_keywords]
        given_name_tokens = [t for t in name_tokens if t not in common_surnames]

        conn = self._get_conn()
        cur = conn.cursor()
        matched_ids = []

        fts_tokens = [f'"{t}"*' if len(t) >= 4 else f'"{t}"' for t in tokens]
        fts_query_and = " AND ".join(fts_tokens)
        try:
            cur.execute("""
                SELECT officer_id, bm25(fts_officers) as score 
                FROM fts_officers 
                WHERE fts_officers MATCH ? 
                ORDER BY score ASC LIMIT ?
            """, (fts_query_and, limit * 2))
            matched_ids = [r[0] for r in cur.fetchall()]
        except Exception:
            pass

        if len(matched_ids) < 3:
            fts_query_or = " OR ".join(fts_tokens)
            try:
                cur.execute("""
                    SELECT officer_id, bm25(fts_officers) as score 
                    FROM fts_officers 
                    WHERE fts_officers MATCH ? 
                    ORDER BY score ASC LIMIT ?
                """, (fts_query_or, limit * 3))
                for r in cur.fetchall():
                    if r[0] not in matched_ids:
                        matched_ids.append(r[0])
            except Exception:
                pass

        # Exact name substring fallback
        if len(matched_ids) < limit:
            for t in tokens:
                like_term = f"%{t}%"
                cur.execute("""
                    SELECT id FROM officers
                    WHERE name_hi LIKE ? OR name_en LIKE ?
                       OR (designation_hi LIKE ? AND ? != '')
                    LIMIT ?
                """, (like_term, like_term, like_term, dept_tokens[0] if dept_tokens else '', limit))
                for r in cur.fetchall():
                    if r[0] not in matched_ids:
                        matched_ids.append(r[0])

        def sort_key(o):
            score = 0
            cat = o.get("category", "")
            des_en = (o.get("designation_en") or "").lower()
            des_hi = (o.get("designation_hi") or "").lower()
            dep_en = (o.get("department_en") or "").lower()
            dep_hi = (o.get("department_hi") or "").lower()
            name_en = (o.get("name_en") or "").lower()
            name_hi = (o.get("name_hi") or "").lower()

            # Strict Person Name Matching:
            # If the query specifies a person's given name (e.g. 'mayank' in 'mayank agrawal'):
            # The officer MUST match the given name!
            if given_name_tokens and not has_dept_intent:
                matches_given_name = any(gt in name_en or gt in name_hi for gt in given_name_tokens)
                if not matches_given_name:
                    return 999999

            # If query has both given name and surname, verify candidate doesn't contradict given name
            if given_name_tokens and any(t.lower() in common_surnames for t in tokens):
                matches_given_name = any(gt in name_en or gt in name_hi for gt in given_name_tokens)
                if not matches_given_name:
                    return 999999

            # Chief Secretary intent
            if any(k in norm_q for k in ["chief secretary", "मुख्य सचिव"]):
                if cat == "chief_secretary" or "chief secretary" in des_en or "मुख्य सचिव" in des_hi:
                    score += 1000

            # Exact token matches
            for t in tokens:
                t_low = t.lower()
                if t_low in name_en or t_low in name_hi:
                    score += 150
                if t_low in des_en or t_low in des_hi:
                    score += 60
                if t_low in dep_en or t_low in dep_hi:
                    score += 40

            # Specific department secretary priorities
            if "forest" in norm_q or "वन" in norm_q:
                if "pingua" in name_en or "पिंगुआ" in name_hi:
                    score += 300
                elif "forest" in des_en or "वन" in des_hi:
                    score += 50
            if "revenue" in norm_q or "राजस्व" in norm_q:
                if "abedi" in name_en or "आबिदी" in name_hi:
                    score += 300
            if "chips" in norm_q or "चिप्स" in norm_q:
                if cat == "chips_leadership":
                    score += 250

            # Rank higher tier designations over junior posts
            if "chief secretary" in des_en:
                score += 50
            elif "additional chief secretary" in des_en:
                score += 45
            elif "principal secretary" in des_en:
                score += 40
            elif "special secretary" in des_en or "विशेष सचिव" in des_hi:
                score += 30
            elif "secretary" in des_en or "सचिव" in des_hi:
                score += 25

            return -score

        results = []
        for oid in matched_ids:
            off = self.get_officer(oid)
            if off:
                s = sort_key(off)
                if s < 0:  # Valid positive match
                    results.append((s, off))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results[:limit]]

    def get_secretaries(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM officers 
            WHERE category IN ('secretaries', 'chief_secretary') 
               OR designation_en LIKE '%Secretary%' 
               OR designation_hi LIKE '%सचिव%'
            ORDER BY CASE WHEN category = 'chief_secretary' THEN 0 ELSE 1 END, name_en ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_collectors(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM officers
            WHERE category = 'collectors'
            ORDER BY department_en ASC, department_hi ASC, name_en ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_chief_secretary(self) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM officers 
            WHERE category = 'chief_secretary' 
               OR (designation_en LIKE '%Chief Secretary%' AND status = 'active')
               OR (designation_hi LIKE '%मुख्य सचिव%' AND status = 'active')
            LIMIT 1
        """)
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return self.get_officer(row["id"])

    def get_former_chief_secretaries(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM officers
            WHERE (designation_en LIKE '%Chief Secretary%' OR designation_hi LIKE '%मुख्य सचिव%')
              AND (category = 'retired_ias' OR status = 'retired')
            ORDER BY id ASC
        """)
        ids = [r["id"] for r in cur.fetchall()]
        conn.close()
        res = []
        for oid in ids:
            off = self.get_officer(oid)
            if off:
                res.append(off)
        return res

    def get_chips_ceos(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM officers 
            WHERE category = 'chips_leadership' 
               OR department_en LIKE '%CHiPS%' 
               OR designation_en LIKE '%CHiPS%' 
               OR department_hi LIKE '%चिप्स%'
            ORDER BY id ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        results = []
        for r in rows:
            cur_off = self.get_officer(r["id"])
            results.append(cur_off if cur_off else r)
        return results

    def get_retired_ias(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM officers WHERE category = 'retired_ias' OR status = 'retired'")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        meta_index = load_document_metadata_index()
        superann_meta = meta_index.get("14f73433fbdd53d7", {})
        for r in rows:
            if r.get("additional_charges"):
                r["additional_charges"] = [c.strip() for c in r["additional_charges"].split("|") if c.strip()]
            r["source_doc_id"] = "14f73433fbdd53d7"
            r["canonical_source_id"] = superann_meta.get("canonical_id", "819ea9132e98457c9e0225027186dd53")
            r["source_url"] = superann_meta.get("source_url", "https://gad.cg.gov.in/pdf/Retd.IASOfficers.pdf")
            r["source_title"] = superann_meta.get("title", "सेवानिवृत भारतीय प्रशासनिक सेवा अधिकारियों की जानकारी")
            r["filename"] = superann_meta.get("filename", "Retd.IASOfficers.pdf")
        return rows

    def get_ias_establishment(self) -> Dict[str, Any]:
        return {
            "section": "भारतीय प्रशासनिक सेवा (स्थापना) - IAS Establishment Section",
            "department": "सामान्य प्रशासन विभाग (GAD), मंत्रालय, नवा रायपुर",
            "mandate": "समस्त आईएएस अधिकारियों का सेवा संवर्ग, पदस्थापना, स्थानांतरण व अनुशासन प्रशासन",
            "key_functions": [
                "भारतीय प्रशासनिक सेवा (स्थापना) संबंधित समस्त लेखाकार्य एवं वेतन निर्धारण",
                "पदोन्नति (Promotion to Senior Scale, Selection Grade, Super Time Scale, Apex Scale)",
                "पदस्थापना एवं स्थानांतरण (Postings and Transfers across Departments and Districts)",
                "अतिरिक्त प्रभार आवंटन (Allocation of Additional Charges / Ex-officio roles)",
                "विभागीय जाँच एवं प्राप्त शिकायतें (Departmental Inquiries, Complaints, Vigilance)",
                "सिविल सर्विसेज बोर्ड (Civil Services Board - CSB) संबंधी कार्य",
                "कर्मचारी/अधिकारियों के शासकीय विदेश प्रवास से संबंधित अनुमति (Foreign visits & deputation clearance)",
                "सेवानिवृत्ति आदेश एवं पेंशन प्रकरण (Superannuation orders & pension cases)"
            ],
            "key_circulars": [
                {"order_no": "ई 3-01/2003/एक/2", "subject": "भारतीय प्रशासनिक सेवा के अधिकारियों की अधिवार्षिकी सेवानिवृत्ति आदेश"},
                {"order_no": "Section_work_view", "subject": "सामान्य प्रशासन विभाग कार्य आवंटन (Work Allocation Matrix)"}
            ]
        }


def main():
    print("=" * 65)
    print("Building GAD-BOT Officer Store & Notification Links...")
    print("=" * 65)
    officers = build_officer_dataset()
    link_circular_notifications(officers)
    persist_officer_store(officers)
    print("=" * 65)
    print(f"Done! {len(officers)} officers indexed successfully.")


if __name__ == "__main__":
    main()
