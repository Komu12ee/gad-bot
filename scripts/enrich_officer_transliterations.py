import sys, os, sqlite3, re, unicodedata, json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

consonants = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy', 'श्र': 'shr', 'ड़': 'd', 'ढ़': 'dh', 'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'फ़': 'f'
}
vowels = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऑ': 'o'
}
matras = {
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h', '़': '', 'ॉ': 'o'
}

def phonetic_transliterate(w):
    res = ''
    i = 0
    n = len(w)
    while i < n:
        ch2 = w[i:i+2]
        if ch2 in consonants:
            cons = consonants[ch2]
            i += 2
        elif w[i] in consonants:
            cons = consonants[w[i]]
            i += 1
        elif w[i] in vowels:
            res += vowels[w[i]]
            i += 1
            continue
        elif w[i] in matras:
            res += matras[w[i]]
            i += 1
            continue
        else:
            res += w[i]
            i += 1
            continue

        if i < n and w[i] in matras:
            res += cons
            if w[i] != '्':
                res += matras[w[i]]
            i += 1
        elif i < n and w[i] in vowels:
            res += cons
        else:
            if i < n and (w[i] in consonants or w[i] in vowels):
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
    # Hindi initials
    'सी': 'C.', 'आर': 'R.', 'ओ': 'O.', 'पी': 'P.', 'डी': 'D.', 'एस': 'S.', 'के': 'K.',
    'एम': 'M.', 'एन': 'N.', 'बी': 'B.', 'ए': 'A.', 'टी': 'T.', 'जे': 'J.', 'वी': 'V.',
    'एच': 'H.', 'जी': 'G.', 'एल': 'L.', 'यू': 'U.'
}

def transliterate_hindi_name(name_hi: str) -> str:
    if not name_hi or not name_hi.strip():
        return ""
    name_clean = re.sub(r'^(श्री|श्रीमती|डॉ\.|डॉ|सुश्री|कु\.|मेसर्स|स्का\.\s*लीडर)\s+', '', name_hi).strip(' ।,')
    
    # Handle compound Hindi initials like के.सी. or पी.एस. or व्ही.एफ.पॉल
    # Add space around dots
    name_clean = re.sub(r'([।\u0900-\u097f])\.', r'\1 . ', name_clean)
    tokens = name_clean.split()
    res = []
    skip_dot = False
    for i, t in enumerate(tokens):
        if t == '.':
            continue
        ct = t.strip(' .,।')
        if not ct:
            continue
        if ct in NAME_DICT:
            val = NAME_DICT[ct]
            res.append(val)
        elif re.match(r'^[A-Za-z\.]+$', t):
            res.append(t)
        else:
            trans = phonetic_transliterate(ct)
            res.append(trans)
    full = ' '.join(res)
    # clean dots formatting like "C. . R." -> "C. R."
    full = re.sub(r'\s*\.\s*', '. ', full)
    full = re.sub(r'\s+', ' ', full).strip()
    return full

DB_PATHS = [
    os.path.join(REPO_ROOT, "gad_kg", "data", "work", "officers.db"),
    os.path.join(REPO_ROOT, "data", "fg2", "temporal_graph.db")
]

for db_path in DB_PATHS:
    if not os.path.exists(db_path):
        continue
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if officers table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='officers';")
    if not cur.fetchone():
        conn.close()
        continue

    # Delete junk rows like id = '1'
    cur.execute("DELETE FROM officers WHERE id = '1' OR name_hi = '1'")
    cur.execute("DELETE FROM officer_notifications WHERE officer_id = '1'")
    try:
        cur.execute("DELETE FROM fts_officers WHERE officer_id = '1'")
    except Exception:
        pass

    cur.execute("SELECT id, name_hi, name_en FROM officers")
    rows = cur.fetchall()
    updated_count = 0

    for oid, name_hi, name_en in rows:
        needs_update = False
        # If name_en contains Hindi Devanagari or is empty
        if not name_en or re.search(r'[\u0900-\u097f]', name_en):
            new_name_en = transliterate_hindi_name(name_hi)
            needs_update = True
        else:
            new_name_en = name_en

        if needs_update:
            cur.execute("UPDATE officers SET name_en = ? WHERE id = ?", (new_name_en, oid))
            updated_count += 1

    conn.commit()

    # Rebuild fts_officers table completely
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_officers';")
    if cur.fetchone():
        cur.execute("DELETE FROM fts_officers;")
        cur.execute("""
            INSERT INTO fts_officers (officer_id, name_hi, name_en, designation_hi, designation_en, department_hi, department_en, additional_charges)
            SELECT id, name_hi, name_en, designation_hi, designation_en, department_hi, department_en, additional_charges
            FROM officers;
        """)
        conn.commit()

    print(f"[{os.path.basename(db_path)}] Processed {len(rows)} officers. Updated {updated_count} names to clean English transliterations. FTS index rebuilt.")

    # Check Devanagari remainder in name_en
    cur.execute("SELECT id, name_hi, name_en FROM officers")
    dev_remain = [r for r in cur.fetchall() if re.search(r'[\u0900-\u097f]', r[2])]
    print(f"[{os.path.basename(db_path)}] Remaining officers with Devanagari in name_en: {len(dev_remain)}")
    if dev_remain:
        for r in dev_remain[:5]:
            print("  ", r)

    conn.close()

print("Enrichment complete.")
