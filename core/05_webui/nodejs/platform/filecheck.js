'use strict';

/**
 * Which files a department may upload, and the checks that a file really is what
 * its name says.
 *
 * The extension picks the type; the bytes must then match it. An .exe renamed
 * report.pdf fails because it does not start with "%PDF-". Office files are opened
 * far enough to see what they hold:
 *   · .docx / .pptx: a zip that must hold the Word or PowerPoint parts, with no
 *     macros (vbaProject.bin) and no "zip bomb" (more than 1 GB when unpacked).
 *   · .doc / .ppt (97-2003): the old container format, which must hold a Word or
 *     PowerPoint stream and no macros (_VBA_PROJECT).
 *   · .txt: UTF-8 text without NUL bytes (a program is not text).
 *   · images: the format's signature.
 * Nothing here runs or unpacks a file to disk. A virus scanner can be added on top
 * (PLATFORM_SCAN_CMD, see routes.js).
 *
 * PLATFORM_FILE_TYPES (comma-separated kinds, e.g. "pdf,docx,txt") narrows the list.
 */

const fs = require('fs/promises');

const TYPES = {
  pdf:  { ext: ['pdf'], label: 'PDF', mime: 'application/pdf' },
  docx: { ext: ['docx'], label: 'Word', mime: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' },
  doc:  { ext: ['doc'], label: 'Word 97-2003', mime: 'application/msword' },
  pptx: { ext: ['pptx'], label: 'PowerPoint', mime: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' },
  ppt:  { ext: ['ppt'], label: 'PowerPoint 97-2003', mime: 'application/vnd.ms-powerpoint' },
  txt:  { ext: ['txt'], label: 'Text', mime: 'text/plain; charset=utf-8' },
  png:  { ext: ['png'], label: 'PNG image', mime: 'image/png' },
  jpg:  { ext: ['jpg', 'jpeg'], label: 'JPEG image', mime: 'image/jpeg' },
  tif:  { ext: ['tif', 'tiff'], label: 'TIFF image', mime: 'image/tiff' },
};

const HEAD_BYTES = 1024;                    // what checkHeader needs to see
const MAX_UNPACKED = 1024 * 1024 * 1024;    // an Office zip may unpack to 1 GB at most
const MAX_ZIP_ENTRIES = 50000;

function allowedKinds() {
  const raw = (process.env.PLATFORM_FILE_TYPES || '').trim().toLowerCase();
  if (!raw) return Object.keys(TYPES);
  const want = raw.split(',').map((s) => s.trim()).filter((k) => TYPES[k]);
  return want.length ? want : Object.keys(TYPES);
}

/** The kind for a file name ("pdf", "docx", …), or null when it may not be uploaded. */
function kindOf(name) {
  const m = /\.([a-z0-9]{1,5})$/i.exec(String(name || ''));
  if (!m) return null;
  const ext = m[1].toLowerCase();
  return allowedKinds().find((k) => TYPES[k].ext.includes(ext)) || null;
}

/** What the upload form may accept: [{kind, label, ext}]. */
function describe() {
  return allowedKinds().map((k) => ({ kind: k, label: TYPES[k].label, ext: TYPES[k].ext }));
}

/** A file name safe to show and store: no folders, control or markup characters. */
function cleanName(name) {
  let s = String(name || '').normalize('NFC');
  s = s.split(/[\\/]/).pop();
  s = s.replace(/[\u0000-\u001f\u007f<>:"|?*]/g, '').replace(/\s+/g, ' ').trim();
  s = s.replace(/^\.+/, '');
  if (s.length > 200) {
    const dot = s.lastIndexOf('.');
    const ext = dot > 0 && s.length - dot <= 6 ? s.slice(dot) : '';
    s = s.slice(0, 200 - ext.length).trim() + ext;
  }
  return s;
}

// ── Signatures ────────────────────────────────────────────────────────────────
const SIG = {
  zip: Buffer.from([0x50, 0x4b, 0x03, 0x04]),
  ole: Buffer.from([0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]),
  png: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  jpg: Buffer.from([0xff, 0xd8, 0xff]),
  tifLE: Buffer.from([0x49, 0x49, 0x2a, 0x00]),
  tifBE: Buffer.from([0x4d, 0x4d, 0x00, 0x2a]),
};
const PROGRAMS = [
  Buffer.from('MZ'),                              // Windows .exe / .dll
  Buffer.from([0x7f, 0x45, 0x4c, 0x46]),          // Linux ELF
  Buffer.from([0xfe, 0xed, 0xfa, 0xce]), Buffer.from([0xfe, 0xed, 0xfa, 0xcf]),
  Buffer.from([0xce, 0xfa, 0xed, 0xfe]), Buffer.from([0xcf, 0xfa, 0xed, 0xfe]),   // macOS Mach-O
  Buffer.from([0xca, 0xfe, 0xba, 0xbe]),          // Mach-O fat binary / Java class
];

const startsWith = (buf, sig) => buf.length >= sig.length && buf.subarray(0, sig.length).equals(sig);

/**
 * Check the first bytes of a file (at least HEAD_BYTES, or the whole file if
 * shorter) against its kind. Returns null when they fit, else a reason to show.
 */
function checkHeader(head, kind) {
  const buf = Buffer.from(head);
  if (!buf.length) return 'The file is empty.';
  if (PROGRAMS.some((sig) => startsWith(buf, sig))) return 'This is a program, not a document.';
  const label = TYPES[kind] ? TYPES[kind].label : kind;
  const wrong = `This file is not a real ${label} file (its contents do not match its name).`;
  switch (kind) {
    case 'pdf':
      return buf.subarray(0, HEAD_BYTES).includes('%PDF-') ? null : wrong;
    case 'docx': case 'pptx':
      return startsWith(buf, SIG.zip) ? null : wrong;
    case 'doc': case 'ppt':
      return startsWith(buf, SIG.ole) ? null : wrong;
    case 'png':
      return startsWith(buf, SIG.png) ? null : wrong;
    case 'jpg':
      return startsWith(buf, SIG.jpg) ? null : wrong;
    case 'tif':
      return startsWith(buf, SIG.tifLE) || startsWith(buf, SIG.tifBE) ? null : wrong;
    case 'txt':
      return buf.includes(0) ? 'This is not a text file (save it as plain UTF-8 text).' : null;
    default:
      return 'This file type cannot be uploaded.';
  }
}

// ── Whole-file checks ─────────────────────────────────────────────────────────

/** Names (and total unpacked size) of the entries in a zip, from its central directory. */
async function zipEntries(file) {
  const fh = await fs.open(file, 'r');
  try {
    const { size } = await fh.stat();
    const tailLen = Math.min(size, 22 + 65535);
    const tail = Buffer.alloc(tailLen);
    await fh.read(tail, 0, tailLen, size - tailLen);
    let eocd = -1;
    for (let i = tailLen - 22; i >= 0; i--) {
      if (tail.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('damaged');
    const count = tail.readUInt16LE(eocd + 10);
    const cdSize = tail.readUInt32LE(eocd + 12);
    const cdOffset = tail.readUInt32LE(eocd + 16);
    // ZIP64 markers: never needed for files this size — treat as unusual.
    if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) throw new Error('unusual');
    if (count > MAX_ZIP_ENTRIES || cdSize > 32 * 1024 * 1024 || cdOffset + cdSize > size) throw new Error('damaged');
    const cd = Buffer.alloc(cdSize);
    await fh.read(cd, 0, cdSize, cdOffset);
    const names = [];
    let unpacked = 0;
    let p = 0;
    for (let n = 0; n < count; n++) {
      if (p + 46 > cd.length || cd.readUInt32LE(p) !== 0x02014b50) throw new Error('damaged');
      const usize = cd.readUInt32LE(p + 24);
      const nameLen = cd.readUInt16LE(p + 28);
      const extraLen = cd.readUInt16LE(p + 30);
      const commentLen = cd.readUInt16LE(p + 32);
      if (usize === 0xffffffff) throw new Error('unusual');
      names.push(cd.toString('utf8', p + 46, p + 46 + nameLen));
      unpacked += usize;
      p += 46 + nameLen + extraLen + commentLen;
    }
    return { names, unpacked };
  } finally {
    await fh.close();
  }
}

/** Does the file contain any of these byte strings? Read in pieces (files are large). */
async function fileContains(file, needles) {
  const longest = Math.max(...needles.map((b) => b.length));
  const fh = await fs.open(file, 'r');
  try {
    const piece = Buffer.alloc(1024 * 1024 + longest);
    const found = new Set();
    let carry = 0;
    let pos = 0;
    for (;;) {
      const { bytesRead } = await fh.read(piece, carry, piece.length - carry, pos);
      if (!bytesRead) break;
      pos += bytesRead;
      const view = piece.subarray(0, carry + bytesRead);
      needles.forEach((b, i) => { if (!found.has(i) && view.includes(b)) found.add(i); });
      carry = Math.min(longest - 1, view.length);
      view.copy(piece, 0, view.length - carry);
    }
    return needles.map((_, i) => found.has(i));
  } finally {
    await fh.close();
  }
}

const utf16 = (s) => Buffer.from(s, 'utf16le');

async function checkText(file) {
  const fh = await fs.open(file, 'r');
  try {
    const decoder = new TextDecoder('utf-8', { fatal: true });
    const piece = Buffer.alloc(1024 * 1024);
    let pos = 0;
    for (;;) {
      const { bytesRead } = await fh.read(piece, 0, piece.length, pos);
      if (!bytesRead) break;
      pos += bytesRead;
      const view = piece.subarray(0, bytesRead);
      if (view.includes(0)) return 'This is not a text file (save it as plain UTF-8 text).';
      decoder.decode(view, { stream: true });
    }
    decoder.decode();
    return null;
  } catch (_) {
    return 'The text is not UTF-8. Save the file as UTF-8 text and upload it again.';
  } finally {
    await fh.close();
  }
}

/**
 * Check a whole received file against its kind. Returns null when it may be kept,
 * else the reason it is refused.
 */
async function checkFile(file, kind) {
  const fh = await fs.open(file, 'r');
  let head;
  try {
    head = Buffer.alloc(HEAD_BYTES);
    const { bytesRead } = await fh.read(head, 0, HEAD_BYTES, 0);
    head = head.subarray(0, bytesRead);
  } finally {
    await fh.close();
  }
  const bad = checkHeader(head, kind);
  if (bad) return bad;

  if (kind === 'docx' || kind === 'pptx') {
    let zip;
    try { zip = await zipEntries(file); }
    catch (_) { return `This ${TYPES[kind].label} file is damaged or in an unusual format.`; }
    const has = (n) => zip.names.includes(n);
    const main = kind === 'docx' ? 'word/document.xml' : 'ppt/presentation.xml';
    if (!has('[Content_Types].xml') || !has(main)) {
      return `This file is not a real ${TYPES[kind].label} file (its contents do not match its name).`;
    }
    if (zip.names.some((n) => /(^|\/)vbaProject\.bin$/i.test(n))) {
      return 'This file contains macros. Save it without macros and upload it again.';
    }
    if (zip.unpacked > MAX_UNPACKED) return 'This file unpacks to more than 1 GB and cannot be accepted.';
    return null;
  }

  if (kind === 'doc' || kind === 'ppt') {
    const stream = kind === 'doc' ? 'WordDocument' : 'PowerPoint Document';
    const [hasStream, hasMacros] = await fileContains(file, [utf16(stream), utf16('_VBA_PROJECT')]);
    if (!hasStream) return `This file is not a real ${TYPES[kind].label} file (its contents do not match its name).`;
    if (hasMacros) return 'This file contains macros. Save it without macros and upload it again.';
    return null;
  }

  if (kind === 'txt') return checkText(file);
  return null;
}

module.exports = {
  TYPES, HEAD_BYTES, allowedKinds, kindOf, describe, cleanName, checkHeader, checkFile, zipEntries,
};
