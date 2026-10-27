(() => {
  if (window.__dnascapeBrandTheme) return;
  window.__dnascapeBrandTheme = true;
  const attached = new WeakMap();
  const brandColor = value => typeof value === 'string' && value.toLowerCase() === '#b03412';

  function themedStyle(style, accent) {
    if (!style) return style;
    const result = {...style};
    for (const key of ['color', 'borderColor', 'backgroundColor']) {
      if (brandColor(style[key])) result[key] = accent;
    }
    return result;
  }

  function themedOptions(options) {
    const accent = getComputedStyle(document.body).getPropertyValue('--dn-accent').trim() || '#b03412';
    const accentText = getComputedStyle(document.body).getPropertyValue('--dn-accent-text').trim() || '#ffffff';
    const pointer = value => value && ({...value, ...(value.label && {label: {
      ...themedStyle(value.label, accent),
      ...(brandColor(value.label.backgroundColor) ? {color: accentText} : {}),
    }})});
    const result = {...options};
    if (options.tooltip?.axisPointer) result.tooltip = {...options.tooltip, axisPointer: pointer(options.tooltip.axisPointer)};
    if (options.axisPointer) result.axisPointer = pointer(options.axisPointer);
    if (Array.isArray(options.graphic)) result.graphic = options.graphic.map(graphic => {
      if (!graphic.style?.rich?.dot) return graphic;
      const dot = graphic.style.rich.dot;
      return {...graphic, style:{...graphic.style, rich:{...graphic.style.rich,
        dot:{...dot, fill:brandColor(dot.fill)?accent:dot.fill}}}};
    });
    if (Array.isArray(options.color)) result.color = options.color.map(color => brandColor(color) ? accent : color);
    if (Array.isArray(options.series)) {
      result.series = options.series.map(series => ({
        ...series,
        // A data-only gesture patch must not introduce undefined style keys:
        // ECharts interprets those as resets to its default palette and width.
        ...('lineStyle' in series && {lineStyle: themedStyle(series.lineStyle, accent)}),
        ...('itemStyle' in series && {itemStyle: themedStyle(series.itemStyle, accent)}),
        ...('areaStyle' in series && {areaStyle: themedStyle(series.areaStyle, accent)}),
        // Custom bars need the element-level flag too; the series setting
        // alone does not disable their native mouse emphasis.
        ...(series.type === 'custom' && series.emphasis?.disabled && typeof series.renderItem === 'function' && {
          renderItem: (...args) => {
            const item = series.renderItem(...args);
            return item ? {...item, emphasisDisabled: true} : item;
          },
        }),
      }));
    }
    for (const key of ['xAxis', 'yAxis']) {
      if (!options[key]) continue;
      const axis = value => ({
        ...value,
        ...(value.nameTextStyle && {nameTextStyle: themedStyle(value.nameTextStyle, accent)}),
        ...(value.axisLabel && {axisLabel: themedStyle(value.axisLabel, accent)}),
        ...(value.axisLine && {axisLine: {
          ...value.axisLine, ...(value.axisLine.lineStyle && {
            lineStyle: themedStyle(value.axisLine.lineStyle, accent),
          }),
        }}),
        ...(value.axisPointer && {axisPointer: pointer(value.axisPointer)}),
      });
      result[key] = Array.isArray(options[key]) ? options[key].map(axis) : axis(options[key]);
    }
    return result;
  }

  let scheduled = false;
  function refresh() {
    scheduled = false;
    let pending = false;
    const dark = document.body.classList.contains('body--dark');
    const accent = getComputedStyle(document.body).getPropertyValue('--dn-accent').trim();
    const theme = `${dark}|${accent}`;
    const variant = dark ? 'dark' : 'light';
    document.querySelectorAll('img[data-dn-logo]').forEach(logo => {
      const source = logo.getAttribute(`data-dn-logo-${variant}`);
      if (source && logo.getAttribute('src') !== source) logo.setAttribute('src', source);
    });
    const faviconSource = document.body.getAttribute(`data-dn-favicon-${variant}`);
    if (faviconSource) {
      let favicon = document.querySelector('link[rel="icon"]');
      if (!favicon) {
        favicon = document.createElement('link');
        favicon.rel = 'icon';
        document.head.appendChild(favicon);
      }
      if (favicon.getAttribute('href') !== faviconSource) {
        favicon.type = 'image/svg+xml';
        favicon.setAttribute('href', faviconSource);
      }
    }
    document.querySelectorAll('.nicegui-echart').forEach(element => {
      const component = getElement(Number(element.id.slice(1)));
      if (!component?.chart) {
        pending = true;
        return;
      }
      const chart = component.chart;
      const previous = attached.get(chart);
      if (previous === undefined) {
        const setOption = chart.setOption.bind(chart);
        // Only presentation styles are copied; data arrays and stored colour
        // choices retain their original values, including in downloaded code.
        chart.setOption = (options, ...args) => {
          const result = setOption(themedOptions(options), ...args);
          // A redraw replaces temporary symbols. Clear the previous hover
          // selection so the same sample can be highlighted again afterward.
          if (options.series) chart.dispatchAction({type: 'updateAxisPointer', currTrigger: 'leave'});
          return result;
        };
      }
      if (previous !== theme) {
        attached.set(chart, theme);
        component.update_chart();
        component.$emit('brandTheme');
      }
    });
    if (pending) setTimeout(schedule, 20);
  }
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    setTimeout(refresh, 0);
  }
  new MutationObserver(schedule).observe(document.body, {
    attributes: true, attributeFilter: ['class', 'data-dn-logo-light', 'data-dn-logo-dark'],
    childList: true, subtree: true,
  });
  new MutationObserver(schedule).observe(document.body, {
    attributes: true, attributeFilter: ['style', 'data-dn-favicon-light', 'data-dn-favicon-dark'],
  });
  schedule();
})();
