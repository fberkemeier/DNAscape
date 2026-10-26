(async () => {
  const results = [], errors = [];
  window.addEventListener('error', event => errors.push(event.message));
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  // Chrome's dump-dom virtual clock does not advance native compositor
  // frames. Use a deterministic frame queue for gestures in this fixture;
  // the live-app stress test exercises the browser's native scheduler.
  window.requestAnimationFrame = callback => setTimeout(() => callback(performance.now()), 16);
  window.cancelAnimationFrame = clearTimeout;
  const frame = () => new Promise(resolve => requestAnimationFrame(resolve));
  const check = (condition, label) => { if (!condition) throw new Error(label); };
  const close = (a, b) => Math.abs(a - b) <= Math.max(1, Math.abs(a), Math.abs(b)) * 1e-8;
  const convert = value => {
    if (!value || typeof value !== 'object') return;
    for (const key of Object.keys(value)) {
      if (key.startsWith(':')) { value[key.slice(1)] = Function('return (' + value[key] + ')')(); delete value[key]; }
      else convert(value[key]);
    }
  };
  const element = document.getElementById('c1'), events = [];
  const chart = echarts.init(element, null, {renderer: 'canvas'});
  const component = {chart, options: {}, $emit: (name, payload) => events.push({name, payload}),
    update_chart: () => chart.setOption(component.options, {
      notMerge:chart.getOption()?.series?.length !== component.options.series?.length})};
  window.getElement = () => component;
  await pause(50);
  const key = (code, pressed) => window.dispatchEvent(new KeyboardEvent(pressed ? 'keydown' : 'keyup', {code, ctrlKey: pressed, bubbles: true}));
  const bounds = () => {
    const options = chart.getOption();
    return Object.fromEntries(Object.entries(component.options.dnascapeNavigation.axes).map(([key, axis]) => {
      const spec = options[axis.dimension + 'Axis'][axis.index];
      return [key, [spec.min, spec.max]];
    }));
  };
  const span = (range, axis) => axis.log ? Math.log(range[1] / range[0]) : range[1] - range[0];
  const mouse = (type, x, y, ctrl = false) => {
    const box = element.getBoundingClientRect();
    const target = type === 'mousedown' ? element : window;
    const event = new MouseEvent(type, {button: 0, buttons: type === 'mouseup' ? 0 : 1, clientX: box.left + x, clientY: box.top + y, ctrlKey: ctrl, bubbles: true, cancelable: true});
    target.dispatchEvent(event);
    return event;
  };
  const wheel = (x, y, delta, ctrl = false) => {
    const box = element.getBoundingClientRect();
    const event = new WheelEvent('wheel', {clientX: box.left + x, clientY: box.top + y, deltaY: delta, ctrlKey: ctrl, bubbles: true, cancelable: true});
    element.dispatchEvent(event);
    return event.defaultPrevented;
  };
  function bounded() {
    for (const [name, range] of Object.entries(bounds())) {
      const axis = component.options.dnascapeNavigation.axes[name];
      check(range[0] >= axis.min - Math.max(1, Math.abs(axis.min)) * 1e-8, name + ' lower bound');
      check(range[1] <= axis.max + Math.max(1, Math.abs(axis.max)) * 1e-8, name + ' upper bound');
    }
  }
  function visibleAxes() {
    chart.getZr().refreshImmediately();
    const rect = chart.getModel().getComponent('grid', 0).coordinateSystem.getRect();
    for (const [name, axis] of Object.entries(component.options.dnascapeNavigation.axes)) {
      const model = chart.getModel().getComponent(axis.dimension + 'Axis', axis.index);
      const position = model.axis.position;
      const boundary = axis.dimension === 'x'
        ? (position === 'top' ? rect.y : rect.y + rect.height)
        : (position === 'right' ? rect.x + rect.width : rect.x);
      let found = false;
      chart.getViewOfComponentModel(model).group.traverse(item => {
        const dash = item.style?.lineDash;
        if (item.type.toLowerCase() !== 'line' || item.ignore || item.style.lineWidth < 1 || (dash && dash !== 'solid' && dash.length)) return;
        const start = item.transformCoordToGlobal(item.shape.x1, item.shape.y1);
        const end = item.transformCoordToGlobal(item.shape.x2, item.shape.y2);
        const along = axis.dimension === 'x' ? 0 : 1, across = 1 - along;
        const length = axis.dimension === 'x' ? rect.width : rect.height;
        if (Math.abs(start[across] - boundary) < 2 && Math.abs(end[across] - boundary) < 2
            && Math.abs(end[along] - start[along]) >= length - 2) found = true;
      });
      check(found, name + ' must have a solid visible line along the plot boundary after zooming');
    }
  }
  function axisTitles() {
    chart.getZr().refreshImmediately();
    const result = {};
    for (const [key, axis] of Object.entries(component.options.dnascapeNavigation.axes)) {
      const model = chart.getModel().getComponent(axis.dimension + 'Axis', axis.index);
      if (!model.get('name')) continue;
      chart.getViewOfComponentModel(model).group.traverse(item => {
        if (item.type !== 'text' || item.style?.text !== model.get('name')) return;
        const box = item.getBoundingRect();
        result[key] = {
          point: item.transformCoordToGlobal(box.x + box.width / 2, box.y + box.height / 2),
          gap: Number(model.get('nameGap')), font: model.getModel('nameTextStyle').getFont(),
          side: model.axis.position,
        };
      });
      check(result[key], key + ' title disappeared');
    }
    const rect = chart.getModel().getComponent('grid', 0).coordinateSystem.getRect();
    return {titles: result, rect: {x:rect.x, y:rect.y, width:rect.width, height:rect.height}};
  }
  function stableTitles(original) {
    const current = axisTitles();
    const moved = Object.entries(original.titles).filter(([key, before]) => {
      const after = current.titles[key], expected = [...before.point], axis = component.options.dnascapeNavigation.axes[key];
      check(before.font === after.font, key + ' title font changed during navigation');
      if (axis.dimension === 'x') {
        expected[0] += current.rect.x + current.rect.width / 2 - original.rect.x - original.rect.width / 2;
      } else {
        check(after.gap >= before.gap, key + ' title spacing shrank during navigation');
        const direction = before.side === 'right' ? 1 : -1;
        const boundary = r => r.x + (direction > 0 ? r.width : 0);
        expected[0] += boundary(current.rect) - boundary(original.rect) + direction * (after.gap - before.gap);
      }
      return !after.point.every((value, index) => close(value, expected[index]));
    });
    check(!moved.length, 'Titles moved during navigation: ' + JSON.stringify({before:original, after:current}));
  }
  async function pixelsAround(position) {
    chart.getZr().refreshImmediately();
    const image = new Image();
    await new Promise(resolve => { image.onload = resolve; image.src = chart.getDataURL({type:'png',pixelRatio:1}); });
    const canvas = document.createElement('canvas'); canvas.width=image.width; canvas.height=image.height;
    const context=canvas.getContext('2d'); context.drawImage(image,0,0);
    return context.getImageData(Math.round(position[0])-2,Math.round(position[1])-2,5,5).data;
  }
  for (const item of window.navigationCases) {
    try {
      const previousCanvas = element.querySelector('canvas');
      previousCanvas?.dispatchEvent(new PointerEvent('pointerout', {pointerType:'mouse', relatedTarget:document.body, bubbles:true}));
      previousCanvas?.dispatchEvent(new MouseEvent('mouseout', {relatedTarget:document.body, bubbles:true}));
      component.options = structuredClone(item.options);
      convert(component.options);
      chart.setOption(component.options, {notMerge: !['line hover off','line hover again'].includes(item.name)});
      await pause(30);
      const originalData = JSON.stringify(chart.getOption().series.map(series => series.data));
      const axes = component.options.dnascapeNavigation.axes;
      const rect = chart.getModel().getComponent('grid', 0).coordinateSystem.getRect();
      const cx = rect.x + rect.width / 2, cy = rect.y + rect.height / 2;
      const original = bounds();
      const originalTitles = axisTitles();
      if (item.name.includes('retained') && item.name.includes('heatmap')) {
        const pixel = chart.convertToPixel({xAxisIndex:0,yAxisIndex:0},[600,300]);
        const rgba=await pixelsAround(pixel), alpha=[...rgba].filter((_value,index)=>index%4===3);
        check(alpha[12]<=2, 'A native NaN must not reveal the coarse raster beneath its tile: '+JSON.stringify(alpha));
      }
      if (item.name === 'unrounded endpoint') {
        const data = chart.getOption().series[0].data, last = data[data.length - 1];
        check(original.x[1] === last[0], 'Automatic X range must end exactly at the final data coordinate');
        check(close(chart.convertToPixel({seriesIndex:0}, last)[0], rect.x + rect.width), 'Last point must meet the right plot edge');
        const labels = chart.getOption().xAxis[0].axisLabel.customValues;
        check(labels.at(-1) === 200000 && !labels.includes(last[0]), 'Irregular data endpoint must not become a forced tick');
      }
      if (component.options.dnascapePlotType === 'Histogram' && !item.name.includes('KDE')) {
        const bars = [];
        chart.getModel().eachSeries(series => {
          if (series.subType !== 'custom') return;
          const data = series.getData();
          data.each(index => {
            const bar = data.getItemGraphicEl(index);
            if (bar?.shape?.width > 1 && bar.shape.height > 1) bars.push({series:series.seriesIndex, index, bar});
          });
        });
        const highlights = () => bars.filter(({bar}) => bar.hoverState === 2)
          .map(({series,index}) => [series,index]);
        const canvas = element.querySelector('canvas'), box = element.getBoundingClientRect();
        const hoverBar = pixel => {
          canvas.dispatchEvent(new PointerEvent('pointermove', {pointerType:'mouse', isPrimary:true, pointerId:1,
            clientX:box.left+pixel[0], clientY:box.top+pixel[1], bubbles:true}));
          canvas.dispatchEvent(new MouseEvent('mousemove', {
            clientX:box.left+pixel[0], clientY:box.top+pixel[1], bubbles:true,
          }));
        };
        // Clear ZRender's cached target before entering a newly drawn view.
        hoverBar([1, 1]);
        const samples = [...new Set(bars.map(bar => bar.series))]
          .flatMap(series => bars.filter(bar => bar.series === series).slice(1, 4));
        for (const target of samples) {
          for (const fraction of [0.15, 0.49, 0.51, 0.85]) {
            const shape = target.bar.shape;
            const pixel = target.bar.transformCoordToGlobal(shape.x + shape.width * fraction, shape.y + shape.height * 0.75);
            const hit = chart.getZr().handler.findHover(...pixel).target;
            const hovered = bars.find(({bar}) => bar === hit);
            check(hovered, 'Probe point must land on a visible histogram bar');
            hoverBar(pixel);
            await pause(25);
            chart.getZr().refreshImmediately();
            const expected = item.name.endsWith('hover off') ? [] : [[hovered.series,hovered.index]];
            check(JSON.stringify(highlights()) === JSON.stringify(expected),
              'Only the hovered histogram bar may highlight: ' + JSON.stringify({expected,actual:highlights(),fraction}));
          }
        }
        canvas.dispatchEvent(new PointerEvent('pointerout', {pointerType:'mouse', relatedTarget:document.body, bubbles:true}));
        canvas.dispatchEvent(new MouseEvent('mouseout', {relatedTarget:document.body, bubbles:true}));
        await pause(25);
        chart.getZr().refreshImmediately();
        check(!highlights().length, 'Histogram highlight remained after leaving the plot');
      }
      if (['line', 'line hover again', 'hovered curves', 'heatmap', 'line hover off', 'heatmap hover off'].includes(item.name)) {
        const disabled = item.name.endsWith('off'), heatmap = item.name.startsWith('heatmap');
        const dataSeries = heatmap ? 1 : 0, index = 4;
        const row = component.options.series[dataSeries].data[index];
        const pixel = chart.convertToPixel({seriesIndex:dataSeries}, row.slice(0,2));
        const box = element.getBoundingClientRect(), canvas = element.querySelector('canvas');
        canvas.dispatchEvent(new PointerEvent('pointermove', {pointerType:'mouse', isPrimary:true, pointerId:1, clientX:box.left+pixel[0], clientY:box.top+pixel[1], bubbles:true}));
        canvas.dispatchEvent(new MouseEvent('mousemove', {clientX:box.left+pixel[0], clientY:box.top+pixel[1], bubbles:true}));
        await pause(40);
        chart.getZr().refreshImmediately();
        const display = chart.getZr().storage.getDisplayList(true);
        const pointers = display.filter(item => item.type.toLowerCase() === 'line' && item.style.stroke === '#b8b0aa');
        check(pointers.length === (disabled ? 0 : heatmap ? 2 : 1), 'Wrong number of hover guides: ' + pointers.length);
        if (!disabled) {
          check(pointers.every(item => item.style.lineDash?.length), 'Hover guides must be dashed');
          check(display.some(item => item.type==='rect' && item.style.fill==='#b03412'), 'Axis value boxes must use DNAscape red');
        }
        if (!heatmap) {
          component.options.series.forEach((series,i) => {
            const marker = chart.getModel().getSeriesByIndex(i).getData().getItemGraphicEl(index);
            check(disabled ? !marker : !!marker, 'Curve hover marker visibility is wrong');
            if (marker) check(marker.getSymbolPath().style.fill === series.lineStyle.color, 'Hover marker must match its curve');
          });
        }
        canvas.dispatchEvent(new PointerEvent('pointerout', {pointerType:'mouse', relatedTarget:document.body, bubbles:true}));
        canvas.dispatchEvent(new MouseEvent('mouseout', {relatedTarget:document.body, bubbles:true}));
        await pause(20);
      }
      chart.getModel().eachComponent('legend', model => {
        if (!model.get('show')) return;
        const group = chart.getViewOfComponentModel(model).group, box = group.getBoundingRect();
        const center = group.transformCoordToGlobal(box.x + box.width / 2, box.y + box.height / 2);
        check(!mouse('mousedown', ...center).defaultPrevented, 'Legend clicks must remain available');
        mouse('mouseup', ...center);
      });
      check(!wheel(cx, cy, -120), 'Ordinary wheel must scroll the page');
      key('ControlRight', true);
      check(!wheel(cx, cy, -120, true), 'Right Ctrl must not zoom the plot');
      key('ControlRight', false);
      check(JSON.stringify(bounds()) === JSON.stringify(original), 'Wheel without left Ctrl changed the plot');
      key('ControlLeft', true);
      check(wheel(cx, cy, -300, true), 'Left Ctrl wheel must prevent browser zoom');
      await frame();
      const zoomed = bounds();
      if (['line', 'heatmap', 'histogram'].includes(item.name)) {
        const response = JSON.parse(JSON.stringify(item.options));
        convert(response);
        component.options = response;
        component.update_chart();
        check(JSON.stringify(bounds()) === JSON.stringify(zoomed), 'A delayed backend response reset a newer viewport');
      }
      stableTitles(originalTitles);
      if (item.name === 'unrounded endpoint') {
        const labels = chart.getOption().xAxis[0].axisLabel.customValues;
        check(labels?.length && labels[0] > zoomed.x[0] && labels.at(-1) < zoomed.x[1], 'Zoomed endpoints must not become forced ticks');
      }
      for (const [name, range] of Object.entries(zoomed)) check(span(range, axes[name]) < span(original[name], axes[name]), name + ' did not zoom');
      visibleAxes();
      key('ControlLeft', false);
      const beforeClick = bounds();
      mouse('mousedown', cx, cy);
      mouse('mousemove', cx + 1, cy + 1);
      mouse('mouseup', cx + 1, cy + 1);
      check(JSON.stringify(bounds()) === JSON.stringify(beforeClick), 'A click with tiny mouse jitter panned the chart');
      mouse('mousedown', cx, cy);
      mouse('mousemove', cx + 30, cy + 10);
      await frame();
      mouse('mouseup', cx, cy);
      check(JSON.stringify(bounds()) === JSON.stringify(beforeClick), 'A drag returned to its start but retained the old displacement');
      mouse('mousedown', cx, cy);
      mouse('mousemove', cx + 35, cy + 15);
      await frame();
      const moving = bounds();
      component.update_chart();
      check(JSON.stringify(bounds()) === JSON.stringify(moving), 'A redraw interrupted an active drag');
      mouse('mousemove', cx + 10000, cy + 10000);
      mouse('mouseup', cx + 10000, cy + 10000);
      const panned = bounds();
      if (item.name === 'line') {
        let navigated = 0;
        const link = document.createElement('button');
        link.textContent = 'Overview';
        link.addEventListener('click', () => navigated++);
        document.body.appendChild(link);
        const ghost = new MouseEvent('click', {bubbles:true,cancelable:true});
        link.dispatchEvent(ghost);
        check(ghost.defaultPrevented && navigated === 0, 'A completed drag leaked a navigation click');
        link.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true}));
        link.click();
        check(navigated === 1, 'A deliberate subsequent navigation click was blocked');
        link.remove();
      }
      stableTitles(originalTitles);
      if (item.name.startsWith('retained')) {
        // There is deliberately no backend response in this probe. Pan from
        // the loaded tile to an unrendered end and inspect existing pixels.
        await pause(30);
        if (item.name.includes('heatmap')) {
          const area=chart.getModel().getComponent('grid',0).coordinateSystem.getRect();
          const rgba=await pixelsAround([area.x+area.width/2,area.y+area.height/2]);
          check([...rgba].some((value,index)=>index%4===3&&value>0), 'The heatmap went blank while waiting for refinement');
        } else {
          const series=chart.getOption().series, index=series.findIndex(s=>s.type==='line');
          const available=series[index].data.filter(row=>row[0]>=panned.x[0]&&row[0]<=panned.x[1]);
          check(available.length>0, 'No coarse points available immediately after panning');
          const rgba=await pixelsAround(chart.convertToPixel({seriesIndex:index},available[Math.floor(available.length/2)]));
          check([...rgba].some((value,index)=>index%4===3&&value>0), 'The curve went blank while waiting for refinement');
        }
      }
      for (const [name, range] of Object.entries(panned)) check(close(span(range, axes[name]), span(zoomed[name], axes[name])), name + ' pan changed the span');
      bounded();
      visibleAxes();
      key('ControlLeft', true);
      if (Object.values(axes).some(axis => axis.dimension === 'x')) {
        const selection = chart.getModel().getComponent('grid', 0).coordinateSystem.getRect();
        const selectionY = selection.y + selection.height / 2;
        const startEvent = mouse('mousedown', selection.x + selection.width / 4, selectionY, true);
        const endEvent = mouse('mousemove', selection.x + selection.width * 3 / 4, selectionY, true);
        const selectedPixels = Math.abs(endEvent.clientX - startEvent.clientX);
        const overlay = element.querySelector('.dn-plot-range-selection');
        check(overlay && close(parseFloat(overlay.style.width), selectedPixels), 'Missing selection highlight');
        check(getComputedStyle(overlay).backgroundColor === 'rgba(176, 52, 18, 0.16)', 'Selection should be shaded red');
        mouse('mouseup', selection.x + selection.width * 3 / 4, selectionY, true);
        check(!element.querySelector('.dn-plot-range-selection'), 'Selection overlay was not removed');
        for (const [name, range] of Object.entries(bounds())) {
          const expected = axes[name].dimension === 'x' ? selectedPixels / selection.width : 1;
          check(close(span(range, axes[name]) / span(panned[name], axes[name]), expected), name + ' region selection changed the wrong dimension');
        }
      }
      for (let repeat = 0; repeat < 8; repeat++) wheel(cx, cy, 1000, true);
      await frame();
      for (const [name, range] of Object.entries(bounds())) check(close(range[0], axes[name].min) && close(range[1], axes[name].max), name + ' zoom out exceeded or missed defaults');
      bounded();
      key('ControlLeft', false);
      const tooltip = component.options.tooltip.formatter;
      if (item.name === 'line') {
        const html = tooltip([{axisValue: 373, value: [373, 8.47], seriesName: 'Replication timing', seriesType: 'line', marker: '<i style="color:red"></i>'}]);
        check(!html.includes('373') && !html.includes('Replication timing') && html.includes('color:red') && html.includes('8.47'), 'Floating hover should contain only coloured Y values');
      }
      if (item.name === 'boxplot') {
        const html = tooltip({value:[1,2,3,4,5],seriesName:'Profile',seriesType:'boxplot',marker:'<i style="color:red"></i>'});
        check(!html.includes('Profile') && /Median: <b>3(?:\.00)?<\/b>/.test(html) && /Max: <b>5(?:\.00)?<\/b>/.test(html), 'Boxplot hovering must preserve all five summary values: ' + html);
      }
      if (item.name === 'independent Boxplot') {
        chart.getModel().getSeries().forEach((series, index) => {
          const values = series.getData().getRawDataItem(0);
          const rect = series.getData().getItemGraphicEl(0)?.children().find(shape => shape.type === 'rect');
          check(!!rect, 'Each numeric box must render');
          const center = rect.transformCoordToGlobal(rect.shape.x + rect.shape.width / 2, rect.shape.y)[0];
          const expected = chart.convertToPixel({seriesIndex: index}, [values[0], values[3]])[0];
          check(Math.abs(center - expected) < .5, 'Box is offset from its X tick');
        });
        const ticks = chart.getOption().xAxis[0].axisTick.customValues;
        check(ticks.includes(1) && ticks.includes(2), 'Box ticks must survive zooming');
      }
      if (item.name === 'grouped histogram guides') {
        const guide = component.options.dnascapeHistogramGuides;
        const graphic = guide.choices[0].graphic;
        const gx = graphic.left ?? chart.getWidth() - graphic.right - guide.width;
        const gy = graphic.top ?? chart.getHeight() - graphic.bottom - guide.height;
        const clickCanvas = (x, y) => {
          const box = element.getBoundingClientRect(), canvas = element.querySelector('canvas');
          canvas.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true}));
          for (const type of ['mousemove', 'mousedown', 'mouseup', 'click']) canvas.dispatchEvent(new MouseEvent(type, {
            clientX:box.left+x, clientY:box.top+y, button:0, buttons:type==='mousedown'?1:0, bubbles:true, cancelable:true,
          }));
        };
        clickCanvas(gx + 30, gy + 15);
        const menu = document.querySelector('.dn-histogram-dataset-menu');
        check(menu?.querySelectorAll('button').length === 2, 'Histogram header must open the dataset menu');
        menu.querySelectorAll('button')[1].click();
        check(guide.selected === 1 && chart.getOption().graphic[0].elements[0].style.text.includes('Second'), 'Menu did not select the second dataset');
        const series = chart.getModel().getSeriesByIndex(0), data = series.getData();
        const bar = data.getItemGraphicEl(0), shape = bar.getBoundingRect();
        const center = bar.transformCoordToGlobal(shape.x + shape.width / 2, shape.y + shape.height / 2);
        clickCanvas(...center);
        check(guide.selected === 0 && events.some(event => event.name === 'plotDataset' && event.payload.index === 0), 'Clicking a histogram bin must select its dataset');
        const otherBar = chart.getModel().getSeriesByIndex(1).getData().getItemGraphicEl(0);
        check(shape.width < rect.width / component.options.series[0].data.length / 2 + 1, 'Grouped bins must share one bin slot');
        check(bar.shape.x + bar.shape.width <= otherBar.shape.x + 0.1, 'Grouped bins must sit beside each other');
      }
      const beforeCancel = bounds();
      mouse('mousedown', cx, cy);
      mouse('mousemove', cx + 50, cy + 50);
      key('Escape', true);
      check(JSON.stringify(bounds()) === JSON.stringify(beforeCancel), 'Escape did not cancel dragging');
      check(originalData === JSON.stringify(chart.getOption().series.map(series => series.data)), 'Navigation modified series data');
      if (item.name === 'line') {
        key('ControlLeft', true);
        const setOption = chart.setOption;
        let paints = 0;
        chart.setOption = (options, ...args) => {
          if (options.xAxis && !options.series) paints++;
          return setOption(options, ...args);
        };
        for (let index = 0; index < 100; index++) wheel(cx, cy, -2, true);
        await frame();
        chart.setOption = setOption;
        check(paints <= 2, 'A wheel burst performed a full redraw for each event');
        key('ControlLeft', false);
      }
      if (item.name === 'wide zoomed ticks') {
        key('ControlLeft', true);
        const beforeWide = axisTitles();
        for (let index = 0; index < 8; index++) wheel(cx, rect.y + rect.height - .1, -500, true);
        await frame();
        const wide = axisTitles();
        check(wide.titles.y.gap > beforeWide.titles.y.gap, 'Wide Y ticks must reserve title clearance');
        stableTitles(beforeWide);
        for (let index = 0; index < 8; index++) wheel(cx, rect.y + rect.height - .1, 500, true);
        await frame();
        stableTitles(wide);
        check(axisTitles().titles.y.gap === wide.titles.y.gap, 'Zooming out must retain the collision gutter');
        key('ControlLeft', false);
      }
      check(chart.getDataURL({type: 'png'}).startsWith('data:image/png'), 'Navigated plot export failed');
      await pause(150);
      check(events.some(event => event.payload?.context === component.options.dnascapeNavigation.context), 'No viewport event was emitted');
      results.push({name: item.name, passed: true, axes: Object.keys(axes)});
    } catch (error) {
      key('ControlLeft', false);
      key('Escape', true);
      results.push({name: item.name, passed: false, error: String(error.stack || error)});
    }
  }
  chart.dispose();
  element.remove();
  await pause(50);
  document.querySelector('#result').textContent = JSON.stringify({results, errors});
})();
