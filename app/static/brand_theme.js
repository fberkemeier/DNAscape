(() => {
  if (window.__dnascapeBrandTheme) return;
  window.__dnascapeBrandTheme = true;
  const attached = new WeakMap();
  const brandColor = value => typeof value === 'string' && value.toLowerCase() === '#b03412';

  function themedStyle(style, dark) {
    if (!dark || !style || !brandColor(style.color)) return style;
    return {...style, color: '#da8068'};
  }

  function themedOptions(options) {
    if (!document.body.classList.contains('body--dark')) return options;
    const result = {...options};
    if (Array.isArray(options.series)) {
      result.series = options.series.map(series => ({
        ...series,
        lineStyle: themedStyle(series.lineStyle, true),
        itemStyle: themedStyle(series.itemStyle, true),
        areaStyle: themedStyle(series.areaStyle, true),
      }));
    }
    for (const key of ['xAxis', 'yAxis']) {
      if (!options[key]) continue;
      const axis = value => ({
        ...value,
        nameTextStyle: themedStyle(value.nameTextStyle, true),
        axisLabel: themedStyle(value.axisLabel, true),
        axisLine: value.axisLine && {
          ...value.axisLine, lineStyle: themedStyle(value.axisLine.lineStyle, true),
        },
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
        chart.setOption = (options, ...args) => setOption(themedOptions(options), ...args);
      }
      if (previous !== dark) {
        attached.set(chart, dark);
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
    attributes: true, attributeFilter: ['class'], childList: true, subtree: true,
  });
  schedule();
})();
