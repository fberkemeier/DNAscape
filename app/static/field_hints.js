(() => {
  if (window.__dnascapeFieldHints) return;
  window.__dnascapeFieldHints = true;
  const labels = window.dnascapeFieldHintLabels || {};
  const selector = '[data-dn-hint],.q-field,.q-checkbox,.q-toggle,.q-radio,.q-slider,.q-uploader,.dn-compact-color-control';
  const tooltipId = 'dn-field-hover-hint';
  const validationFormId = 'dn-field-validation-owner';
  let tooltip, target, timer, hideTimer, described;
  let pointerX = 0, pointerY = 0, tooltipWidth = 0, tooltipHeight = 0;
  const enabled = () => !document.body.classList.contains('dn-field-hints-disabled');

  function useAppValidation(root) {
    const controls = root.matches?.('input,textarea,select') ? [root] : root.querySelectorAll('input,textarea,select');
    for (const control of controls) {
      if (!control.closest('.q-field,[data-dn-hint]') || control.type === 'file' || control.type === 'hidden') continue;
      // Chromium uses validation messages as default hover tooltips unless the
      // field belongs to a novalidate form. Keep min/max/step and validity APIs:
      // DNAscape and Quasar already validate edits and show useful errors.
      let form = control.form;
      if (!form) {
        form = document.getElementById(validationFormId);
        if (!form) {
          form = document.createElement('form');
          form.id = validationFormId;
          form.hidden = true;
          form.noValidate = true;
          form.addEventListener('submit', event => event.preventDefault());
          document.body.appendChild(form);
        }
        control.setAttribute('form', validationFormId);
      }
      form.noValidate = true;
    }
  }

  function positionHint() {
    const left = Math.max(8, Math.min(innerWidth - tooltipWidth - 8, pointerX + 12));
    let top = pointerY + 18;
    if (top + tooltipHeight > innerHeight - 8) top = pointerY - tooltipHeight - 10;
    tooltip.style.left = left + 'px';
    tooltip.style.top = Math.max(8, Math.min(innerHeight - tooltipHeight - 8, top)) + 'px';
  }

  function labelFor(field) {
    return (field.querySelector('.q-field__label,.q-checkbox__label,.q-toggle__label,.q-radio__label,.dn-compact-color-label')?.textContent
      || field.getAttribute('aria-label') || field.querySelector('[aria-label]')?.getAttribute('aria-label') || '')
      .replace(/\s+/g, ' ').trim().replace(/\s*\*$/, '');
  }

  function textFor(field) {
    // Quasar places unknown input attributes on the native input; other
    // controls keep them on their root. Both are the same logical field.
    const explicit = field.dataset.dnHint || field.querySelector('[data-dn-hint]')?.dataset.dnHint
      || field.closest('[data-dn-hint]')?.dataset.dnHint;
    if (explicit) return explicit;
    if (field.matches('.q-uploader')) return labels.Upload;
    const label = labelFor(field);
    if (field.closest('.q-color-picker')) {
      const channels = {
        R: 'Red colour channel.', G: 'Green colour channel.', B: 'Blue colour channel.',
        A: 'Opacity of the selected colour.', ALPHA: 'Opacity of the selected colour.',
        H: 'Hue of the selected colour.', S: 'Saturation of the selected colour.',
        V: 'Brightness of the selected colour.',
        HEX: 'Colour written as a hexadecimal code.', HEXA: 'Colour and opacity written as a hexadecimal code.',
        RGB: 'Colour as red, green and blue values.',
        RGBA: 'Colour as red, green, blue and opacity values.',
      };
      if (channels[label.toUpperCase()]) return channels[label.toUpperCase()];
      if (field.classList.contains('q-slider'))
        return field.closest('.q-color-picker__alpha') ? channels.A : channels.H;
      return 'Enter the selected colour value.';
    }
    if (labels[label]) return labels[label];
    if (/^Source(?: \d+)? \(public file link\)$/i.test(label) || /^Public file link/i.test(label))
      return 'Public URL of the file to load.';
    const bare = label.replace(/\s*\([^)]*\)\s*$/, '').trim();
    if (labels[bare]) return labels[bare];
    if (/^Source(?: \d+)? transform formula$/i.test(label))
      return 'Transform this source using Python / NumPy.';
    if (/^Source \d+$/i.test(label)) return labels.Source;
    if (/^(?:X|Y|Z)(?:2)? (?:min|minimum|max|maximum)$/i.test(label))
      return (/min/i.test(label) ? 'Lowest' : 'Highest') + ' value shown on this axis.';
    if (/^(?:Start|Region start)(?:\s|$)/i.test(label)) return 'First position of the selected region.';
    if (/^(?:End|End boundary|Region end)(?:\s|$)/i.test(label)) return 'End position of the selected region.';
    if (/^Page of \d+$/i.test(label)) return labels.Page;
    if (/^Set /i.test(label)) return 'Use a custom value for this setting.';
    if (!label) return '';
    // Future dynamically named controls still receive a brief, safe fallback.
    return (field.matches('.q-checkbox,.q-toggle,.q-radio') ? 'Enable ' :
      field.classList.contains('q-select') ? 'Choose ' : 'Set ') + label.toLowerCase() + '.';
  }

  function fieldAt(node) {
    if (!(node instanceof Element) || node.closest('.dn-field-help-icon')) return null;
    const action = node.closest('button,.q-btn,a,[role="button"]');
    if (action && !action.closest('.dn-compact-color-control,.q-uploader')) return null;
    const field = node.closest(selector);
    return field?.closest('.q-field') || field;
  }

  function hide(immediate = false) {
    clearTimeout(timer);
    clearTimeout(hideTimer);
    timer = null;
    if (described) {
      const ids = (described.getAttribute('aria-describedby') || '').split(/\s+/).filter(id => id && id !== tooltipId);
      if (ids.length) described.setAttribute('aria-describedby', ids.join(' '));
      else described.removeAttribute('aria-describedby');
    }
    described = null;
    target = null;
    if (tooltip) {
      tooltip.classList.remove('dn-field-hover-hint-visible');
      tooltip.setAttribute('aria-hidden', 'true');
      if (immediate) tooltip.hidden = true;
      else hideTimer = setTimeout(() => { tooltip.hidden = true; }, 300);
    }
  }

  function show(field) {
    if (!enabled() || target !== field || !field.isConnected) return;
    const text = textFor(field);
    if (!text) return;
    clearTimeout(hideTimer);
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = tooltipId;
      tooltip.className = 'dn-field-hover-hint';
      tooltip.setAttribute('role', 'tooltip');
      document.body.appendChild(tooltip);
    }
    tooltip.textContent = text;
    tooltip.hidden = false;
    tooltip.style.left = '0px';
    tooltip.style.top = '0px';
    const rect = field.getBoundingClientRect(), box = tooltip.getBoundingClientRect();
    if (!rect.width || !rect.height) { hide(true); return; }
    tooltipWidth = box.width;
    tooltipHeight = box.height;
    positionHint();
    described = field.querySelector('input,textarea,[role="combobox"]') || field;
    const ids = new Set((described.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
    ids.add(tooltipId);
    described.setAttribute('aria-describedby', [...ids].join(' '));
    tooltip.setAttribute('aria-hidden', 'false');
    tooltip.classList.add('dn-field-hover-hint-visible');
  }

  document.addEventListener('pointerover', event => {
    const field = fieldAt(event.target);
    pointerX = event.clientX;
    pointerY = event.clientY;
    if (field) useAppValidation(field);
    if (field === target) return;
    hide();
    if (!field || !enabled()) return;
    target = field;
    timer = setTimeout(() => show(field), 450);
  }, true);
  document.addEventListener('pointermove', event => {
    if (!target || fieldAt(event.target) !== target) return;
    pointerX = event.clientX;
    pointerY = event.clientY;
    if (tooltip && !tooltip.hidden && tooltip.classList.contains('dn-field-hover-hint-visible')) positionHint();
  }, true);
  document.addEventListener('pointerout', event => {
    if (target && fieldAt(event.relatedTarget) !== target) hide();
  }, true);
  document.addEventListener('pointerdown', () => hide(true), true);
  document.addEventListener('keydown', () => hide(true), true);
  document.addEventListener('invalid', event => {
    if (event.target.closest?.('.q-field,[data-dn-hint]')) event.preventDefault();
  }, true);
  document.addEventListener('scroll', () => hide(true), true);
  window.addEventListener('resize', () => hide(true));
  window.addEventListener('blur', () => hide(true));
  const observe = () => {
    useAppValidation(document);
    new MutationObserver(() => { if (!enabled()) hide(true); }).observe(document.body, {attributes:true, attributeFilter:['class']});
    new MutationObserver(records => {
      if (target && !target.isConnected) hide(true);
      for (const record of records) for (const node of record.addedNodes) {
        if (node instanceof Element) useAppValidation(node);
      }
    }).observe(document.body, {childList:true, subtree:true});
  };
  if (document.body) observe();
  else document.addEventListener('DOMContentLoaded', observe, {once:true});
  // Also exposes deterministic descriptions to the browser accessibility audit.
  window.dnascapeFieldHintText = textFor;
})();
