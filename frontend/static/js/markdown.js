/**
 * Minimal, safe Markdown renderer for chat answers.
 * Supports: **bold**, *italic*, `code`, headings, bullet/numbered lists, tables, paragraphs,
 * and highlights [Source N] citations. All text is HTML-escaped FIRST, so model output can
 * never inject markup.
 */
(function (root) {
  function esc(text) {
    return String(text)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/\[(Source\s+\d+)\]/g, '<span class="cite">[$1]</span>');
  }

  const isTableSep = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line || '');
  const cells = (line) => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());

  function renderMarkdown(text) {
    const lines = esc(text || '').replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }

      if (line.trim().startsWith('|') && isTableSep(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().startsWith('|')) { rows.push(cells(lines[i])); i++; }
        out.push('<div class="md-table"><table><thead><tr>' +
          head.map((c) => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>' +
          rows.map((r) => '<tr>' + r.map((c) => `<td>${inline(c)}</td>`).join('') + '</tr>').join('') +
          '</tbody></table></div>');
        continue;
      }

      const heading = line.match(/^\s*(#{1,4})\s+(.*)$/);
      if (heading) { out.push(`<p class="md-h"><strong>${inline(heading[2])}</strong></p>`); i++; continue; }

      if (/^\s*[-*•]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*[-*•]\s+/.test(lines[i])) {
          items.push(`<li>${inline(lines[i].replace(/^\s*[-*•]\s+/, ''))}</li>`); i++;
        }
        out.push(`<ul>${items.join('')}</ul>`);
        continue;
      }

      if (/^\s*\d+[.)]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          items.push(`<li>${inline(lines[i].replace(/^\s*\d+[.)]\s+/, ''))}</li>`); i++;
        }
        out.push(`<ol>${items.join('')}</ol>`);
        continue;
      }

      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*([-*•]|\d+[.)])\s+/.test(lines[i]) &&
             !lines[i].trim().startsWith('|') && !/^\s*#{1,4}\s+/.test(lines[i])) {
        para.push(inline(lines[i])); i++;
      }
      if (para.length) out.push(`<p>${para.join('<br>')}</p>`);
      else i++;
    }
    return out.join('');
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { renderMarkdown };
  else root.renderMarkdown = renderMarkdown;
})(typeof window !== 'undefined' ? window : globalThis);
