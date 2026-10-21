(() => {
  if (window.__dnascapePlotNavigation) return;
  window.__dnascapePlotNavigation = true;
  const attached = new WeakMap(), live = new Set();
  let leftControl = false, drag = null;
  let datasetMenu = null;
  const clamp = (value, lower, upper) => Math.max(lower, Math.min(value, upper));
  const forward = (value, axis) => axis.log ? Math.log(value) : value;
  const backward = (value, axis) => axis.log ? Math.exp(value) : value;
  const changed = (first, second) => !first || first.some((value, index) =>
    Math.abs(value - second[index]) > Math.max(1, Math.abs(value), Math.abs(second[index])) * 1e-12);

  function bounded(lower, upper, axis) {
    const floor = forward(axis.min, axis), ceiling = forward(axis.max, axis);
    const minimum = Math.max((ceiling - floor) * 1e-9, Math.max(1, Math.abs(floor)) * Number.EPSILON * 16);
    const span = clamp(upper - lower, minimum, ceiling - floor);
    const start = clamp(lower, floor, ceiling - span);
    return [backward(start, axis), backward(start + span, axis)];
  }

  function metadata(state) {
    return state.component.options?.dnascapeNavigation;
  }

  function rectangle(state) {
    return state.chart.getModel().getComponent('grid', 0)?.coordinateSystem?.getRect();
  }

  function point(state, event) {
    const box = state.element.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  }

  function inside(rect, position) {
    return rect && position[0] >= rect.x && position[0] <= rect.x + rect.width &&
      position[1] >= rect.y && position[1] <= rect.y + rect.height;
  }

  function inLegend(state, position) {
    let hit = false;
    state.chart.getModel().eachComponent('legend', model => {
      if (!model.get('show')) return;
      const group = state.chart.getViewOfComponentModel(model)?.group;
      if (group && group.getBoundingRect().contain(...group.transformCoordToLocal(...position))) hit = true;
    });
    return hit;
  }

  function guideBox(state) {
    const guide = state.component.options?.dnascapeHistogramGuides;
    if (!guide?.visible) return null;
    const placement = guide.choices[guide.selected || 0]?.graphic;
    if (!placement) return null;
    return {x:placement.left ?? state.chart.getWidth() - placement.right - guide.width,
      y:placement.top ?? state.chart.getHeight() - placement.bottom - guide.height,
      width:guide.width,height:guide.height};
  }

  function chooseDataset(state, index) {
    const guide = state.component.options?.dnascapeHistogramGuides;
    if (!guide?.choices[index]) return;
    guide.selected = index;
    const graphic = guide.choices[index].graphic;
    state.chart.setOption({graphic:guide.visible?[graphic]:[]},{replaceMerge:['graphic']});
    state.component.$emit('plotDataset',{index});
    datasetMenu?.remove(); datasetMenu = null;
  }

  window.dnascapeHistogramMenu = event => {
    const root = event.event?.target?.closest?.('.dn-plot-navigation');
    const state = [...live].find(item => item.element === root);
    const guide = state?.component.options?.dnascapeHistogramGuides;
    if (!guide || guide.choices.length < 2) return;
    state.chart.dispatchAction({type: 'hideTip'});
    state.chart.dispatchAction({type: 'updateAxisPointer', currTrigger: 'leave'});
    datasetMenu?.remove();
    const menu = document.createElement('div');
    menu.className = 'dn-histogram-dataset-menu';
    menu.setAttribute('role','menu');
    menu.setAttribute('aria-label', 'Histogram dataset');
    const box = guideBox(state), origin = root.getBoundingClientRect();
    menu.style.left = Math.min(innerWidth - 240, Math.max(8,origin.left + box.x)) + 'px';
    menu.style.top = Math.min(innerHeight - 100, Math.max(8,origin.top + box.y + 25)) + 'px';
    guide.choices.forEach((choice,index) => {
      const button = document.createElement('button');
      button.title = choice.name;
      button.type = 'button'; button.setAttribute('role','menuitemradio');
      button.setAttribute('aria-checked',String(index === guide.selected));
      const dot = document.createElement('span'); dot.className='dn-histogram-dataset-dot';
      dot.style.backgroundColor=choice.color?.toLowerCase()==='#b03412' ? getComputedStyle(document.body).getPropertyValue('--dn-accent').trim()||choice.color : choice.color;
      const text = document.createElement('span'); text.textContent=choice.name;
      button.append(dot,text); button.addEventListener('click',() => chooseDataset(state,index));
      menu.appendChild(button);
    });
    document.body.appendChild(menu); datasetMenu=menu;
    menu.style.left = clamp(origin.left + box.x, 8, Math.max(8, innerWidth - menu.offsetWidth - 8)) + 'px';
    menu.style.top = clamp(origin.top + box.y + 25, 8, Math.max(8, innerHeight - menu.offsetHeight - 8)) + 'px';
    menu.querySelector('button')?.focus({preventScroll:true});
  };

  function ranges(state) {
    const result = {};
    for (const [key, axis] of Object.entries(metadata(state)?.axes || {})) {
      const spec = state.chart.getModel().getComponent(axis.dimension + 'Axis', axis.index)?.option;
      const values = [Number(spec?.min), Number(spec?.max)];
      if (values.every(Number.isFinite) && values[1] > values[0] && (!axis.log || values[0] > 0)) result[key] = values;
    }
    return result;
  }

  function valueAt(state, axis, pixel) {
    // Derive each scale independently. ECharts' Cartesian conversion can
    // choose a different axis pair when multiple boxplot axes share a grid.
    const rect = rectangle(state), horizontal = axis.dimension === 'x';
    const spec = state.chart.getModel().getComponent(axis.dimension + 'Axis', axis.index).option;
    let fraction = (pixel - (horizontal ? rect.x : rect.y)) / (horizontal ? rect.width : rect.height);
    if (!horizontal) fraction = 1 - fraction;
    if (axis.inverse) fraction = 1 - fraction;
    return backward(forward(spec.min, axis) + fraction * (forward(spec.max, axis) - forward(spec.min, axis)), axis);
  }

  function flush(state) {
    clearTimeout(state.timer);
    if (!state.pending) return;
    const payload = state.pending;
    state.pending = null;
    if (payload.context === metadata(state)?.context && state.element.isConnected) state.component.$emit('plotViewport', payload);
  }

  function apply(state, next, notify = true) {
    const current = ranges(state), patch = {};
    const updates = Object.fromEntries(Object.entries(next).filter(([key, values]) => changed(current[key], values)));
    if (!Object.keys(updates).length) return;
    for (const [key, values] of Object.entries(updates)) {
      const axis = metadata(state).axes[key], name = axis.dimension + 'Axis';
      if (!patch[name]) {
        const count = Math.max(...Object.values(metadata(state).axes).filter(item => item.dimension === axis.dimension).map(item => item.index)) + 1;
        patch[name] = Array.from({length: count}, () => ({}));
      }
      const spec = patch[name][axis.index];
      Object.assign(spec, {min: values[0], max: values[1], interval: null});
      const axisLine = state.chart.getModel().getComponent(name, axis.index).option.axisLine || {};
      spec.axisLine = {...axisLine, show:true, onZero:false,
        lineStyle:{width:1.2, ...axisLine.lineStyle, type:'solid', opacity:1}};
      for (const part of ['axisLabel', 'axisTick', 'splitLine']) spec[part] = {...spec[part], customValues: null};
      Object.assign(spec.axisLabel, {showMinLabel: true, showMaxLabel: true});
    }
    state.chart.setOption(patch, {lazyUpdate: false, silent: true});
    if (notify) {
      state.pending = {context: metadata(state).context, ranges: {...state.pending?.ranges, ...updates}};
      clearTimeout(state.timer);
      state.timer = setTimeout(() => flush(state), 120);
    }
  }

  function flushFrame(state) {
    if (state.frame === undefined) return;
    cancelAnimationFrame(state.frame);
    state.frame = undefined;
    apply(state, state.next, false);
  }

  function paint(state, next) {
    state.next = next;
    if (state.frame === undefined) state.frame = requestAnimationFrame(() => flushFrame(state));
  }

  function cursor(state, position) {
    if (drag?.state === state) state.chart.getZr().setCursorStyle(drag.mode === 'select' ? 'crosshair' : 'grabbing');
    else if (inLegend(state, position) || inside(guideBox(state),position)) state.chart.getZr().setCursorStyle('pointer');
    else state.chart.getZr().setCursorStyle(inside(rectangle(state), position) ? (leftControl ? 'crosshair' : 'grab') : 'default');
  }

  function down(state, event) {
    if (event.button !== 0) return;
    const rect = rectangle(state), start = point(state, event), initial = ranges(state);
    if (!inside(rect, start) || inLegend(state, start) || inside(guideBox(state),start) || !Object.keys(initial).length) return;
    flush(state);
    const mode = leftControl && event.ctrlKey ? 'select' : 'pan';
    if (mode === 'select' && !Object.keys(initial).some(key => key.endsWith('x'))) return;
    event.preventDefault();
    state.suppressClickUntil = 0;
    if (mode === 'select') event.stopImmediatePropagation();
    drag = {state, mode, start, end: start, rect, initial, context: metadata(state).context};
    if (mode === 'select') {
      const overlay = document.createElement('div');
      overlay.className = 'dn-plot-range-selection';
      overlay.setAttribute('aria-hidden', 'true');
      overlay.style.cssText = `left:${start[0]}px;top:${rect.y}px;width:0;height:${rect.height}px;`;
      state.element.appendChild(overlay);
      drag.overlay = overlay;
    }
    cursor(state, start);
  }

  function move(event) {
    if (!drag) return;
    const {state, rect, initial, start} = drag;
    if (!state.element.isConnected || drag.context !== metadata(state)?.context) return cancel();
    event.preventDefault();
    event.stopImmediatePropagation();
    drag.end = point(state, event);
    if (drag.mode === 'select') {
      const end = clamp(drag.end[0], rect.x, rect.x + rect.width);
      drag.overlay.style.left = Math.min(start[0], end) + 'px';
      drag.overlay.style.width = Math.abs(end - start[0]) + 'px';
    } else {
      const next = {};
      for (const [key, values] of Object.entries(initial)) {
        const axis = metadata(state).axes[key], horizontal = axis.dimension === 'x';
        const delta = drag.end[horizontal ? 0 : 1] - start[horizontal ? 0 : 1];
        const lower = forward(values[0], axis), upper = forward(values[1], axis);
        const shift = delta / (horizontal ? rect.width : rect.height) * (upper - lower) *
          (horizontal ? -1 : 1) * (axis.inverse ? -1 : 1);
        next[key] = bounded(lower + shift, upper + shift, axis);
      }
      paint(state, next);
    }
    cursor(state, drag.end);
  }

  function up(event) {
    if (!drag || event.button !== 0) return;
    const pointAtUp = point(drag.state,event);
    if (drag.mode === 'pan' && Math.hypot(pointAtUp[0]-drag.start[0],pointAtUp[1]-drag.start[1]) < 3) {
      const state=drag.state; drag=null; flushFrame(state); cursor(state,pointAtUp); return;
    }
    move(event);
    if (!drag) return;
    const {state, initial, mode, start, end, rect} = drag;
    state.suppressClickUntil = Date.now() + 250;
    flushFrame(state);
    if (mode === 'select' && Math.abs(end[0] - start[0]) >= 3) {
      const next = {};
      for (const [key, axis] of Object.entries(metadata(state).axes)) {
        if (axis.dimension !== 'x') continue;
        const values = [start[0], clamp(end[0], rect.x, rect.x + rect.width)]
          .map(pixel => valueAt(state, axis, pixel)).sort((a, b) => a - b);
        if (values.every(Number.isFinite)) next[key] = bounded(forward(values[0], axis), forward(values[1], axis), axis);
      }
      apply(state, next);
    } else if (mode === 'pan') {
      const current = ranges(state);
      const moved = Object.fromEntries(Object.entries(current).filter(([key, values]) => changed(initial[key], values)));
      if (Object.keys(moved).length) state.pending = {context: metadata(state).context, ranges: moved};
    }
    drag.overlay?.remove();
    drag = null;
    flush(state);
    cursor(state, end);
  }

  function cancel() {
    if (!drag) return;
    const {state, initial} = drag;
    if (state.frame !== undefined) cancelAnimationFrame(state.frame);
    state.frame = undefined;
    if (state.element.isConnected && !state.chart.isDisposed() && drag.context === metadata(state)?.context) apply(state, initial, false);
    drag.overlay?.remove();
    drag = null;
    if (!state.chart.isDisposed()) state.chart.getZr().setCursorStyle('default');
  }

  function wheel(state, event) {
    if (!leftControl || !event.ctrlKey || drag) return;
    const position = point(state, event), rect = rectangle(state);
    if (!inside(rect, position)) return;
    const axes = metadata(state)?.axes;
    if (!axes || !Object.keys(axes).length) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1);
    const factor = Math.exp(clamp(pixels * 0.002, -1, 1)), next = {};
    for (const [key, values] of Object.entries(ranges(state))) {
      const axis = axes[key], lower = forward(values[0], axis), upper = forward(values[1], axis);
      const value = valueAt(state, axis, position[axis.dimension === 'x' ? 0 : 1]);
      const anchor = clamp(forward(value, axis), lower, upper);
      if (!Number.isFinite(anchor)) continue;
      next[key] = bounded(anchor - (anchor - lower) * factor, anchor + (upper - anchor) * factor, axis);
    }
    apply(state, next);
  }

  let scheduled = false;
  function refresh() {
    scheduled = false;
    for (const state of live) {
      if (state.element.isConnected && !state.chart.isDisposed()) continue;
      if (drag?.state === state) cancel();
      state.dispose();
      live.delete(state);
    }
    let pending = false;
    document.querySelectorAll('.nicegui-echart.dn-plot-navigation').forEach(element => {
      const component = getElement(Number(element.id.slice(1))), chart = component?.chart;
      if (!chart) { pending = true; return; }
      if (attached.has(chart)) return;
      const state = {element, component, chart};
      const onClick = event => {
        if (state.suppressClickUntil > Date.now()) return;
        if (component.options?.dnascapeHistogramGuides && event.componentType === 'series') {
          const series=component.options.series?.[event.seriesIndex];
          chooseDataset(state,series?.dnascapeDataset ?? event.seriesIndex);
        }
      };
      chart.on('click',onClick);
      const onDown = event => down(state, event), onWheel = event => wheel(state, event);
      const onMove = event => { if (!drag) { state.lastPoint = point(state, event); cursor(state, state.lastPoint); } };
      element.addEventListener('mousedown', onDown, true);
      element.addEventListener('wheel', onWheel, {capture: true, passive: false});
      element.addEventListener('mousemove', onMove);
      state.dispose = () => {
        clearTimeout(state.timer);
        chart.off('click',onClick);
        if (state.frame !== undefined) cancelAnimationFrame(state.frame);
        element.removeEventListener('mousedown', onDown, true);
        element.removeEventListener('wheel', onWheel, true);
        element.removeEventListener('mousemove', onMove);
      };
      attached.set(chart, state);
      live.add(state);
    });
    if (pending) setTimeout(schedule, 20);
  }
  function schedule() {
    if (!scheduled) { scheduled = true; setTimeout(refresh, 0); }
  }
  function key(event, pressed) {
    if (event.code === 'ControlLeft') leftControl = pressed;
    if (event.code === 'Escape' && pressed) cancel();
    if (event.code === 'Escape' && pressed) { datasetMenu?.remove(); datasetMenu=null; }
    for (const state of live) if (state.lastPoint) cursor(state, state.lastPoint);
  }
  window.addEventListener('keydown', event => key(event, true), true);
  window.addEventListener('keyup', event => key(event, false), true);
  window.addEventListener('blur', () => { leftControl = false; cancel(); });
  window.addEventListener('resize', cancel);
  window.addEventListener('mousemove', move, true);
  window.addEventListener('mouseup', up, true);
  // Flush before toolbar clicks so SVG/PDF downloads use the displayed window.
  document.addEventListener('pointerdown', event => {
    for (const state of live) flush(state);
    if (datasetMenu && !datasetMenu.contains(event.target)) { datasetMenu.remove(); datasetMenu=null; }
  }, true);
  new MutationObserver(schedule).observe(document.body, {childList: true, subtree: true});
  schedule();
})();
