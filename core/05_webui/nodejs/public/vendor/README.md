# Markdown reader assets

Pinned browser builds served locally, with no runtime CDN or npm install required:

- Marked 18.1.0 (MIT): https://cdn.jsdelivr.net/npm/marked@18.1.0/lib/marked.umd.js
- DOMPurify 3.4.16 (Apache 2.0 or MPL 2.0): https://raw.githubusercontent.com/cure53/DOMPurify/3.4.16/dist/purify.min.js

Upstream licenses are included beside the builds. Document HTML is sanitized with
an explicit allowlist before display; HTML tables retain merged cells. Do not
render Marked output directly without sanitizing it.
