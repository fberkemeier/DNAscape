(() => {
  if (window.__dnascapePlotNavigation) return;
  window.__dnascapePlotNavigation = true;
  const attached = new WeakMap(), live = new Set();
  let leftControl = false, drag = null, sequence = 0, suppressNavigationClickUntil = 0;
  let datasetMenu = null;
  const clamp = (value, lower, upper) => Math.max(lower, Math.min(value, upper));
  const forward = (value, axis) => axis.log ? Math.log(value) : value;
  const backward = (value, axis) => axis.log ? Math.exp(value) : value;
  const changed = (first, second) => !first || first.some((value, index) =>
    Math.abs(value - second[index]) > Math.max(1, Math.abs(value), Math.abs(second[index])) * 1e-12);

  function horizontalTicks(lower, upper) {
    // Match the server's aligned 1/2/2.5/5 steps during live mouse gestures.
    // Explicit viewport edges need not be ticks, especially after dragging.
    const rough = (upper - lower) / 6, magnitude = 10 ** Math.floor(Math.log10(rough));
    const fraction = rough / magnitude;
    const step = [1, 2, 2.5, 5, 10].find(value => fraction <= value) * magnitude;
    if (!(step > 0) || !Number.isFinite(step)) return [];
    const tolerance = step * 1e-9, first = Math.ceil((lower - tolerance) / step) * step;
    const ticks = [];
    for (let index = 0; index < 128; index++) {
      const tick = first + index * step;
      if (tick > upper + tolerance) break;
      if (tick >= lower - tolerance) ticks.push(Math.abs(tick) <= tolerance ? 0 : Number(tick.toPrecision(12)));
    }
    return ticks;
  }

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
    if (metadata(state)?.polar) {
      const polar = state.chart.getModel().getComponent('polar', 0)?.coordinateSystem;
      if (polar) {
        const count = state.chart.getOption().polar.length;
        const outer = state.chart.getModel().getComponent('polar', count - 1).coordinateSystem;
        const radius = Math.max(...outer.getRadiusAxis().getExtent());
        return {x:polar.cx-radius,y:polar.cy-radius,width:2*radius,height:2*radius,
          cx:polar.cx,cy:polar.cy,radius,polar:true};
      }
    }
    return state.chart.getModel().getComponent('grid', 0)?.coordinateSystem?.getRect();
  }

  function point(state, event) {
    const box = state.element.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  }

  function inside(rect, position) {
    if (rect?.polar) return Math.hypot(position[0]-rect.cx,position[1]-rect.cy) <= rect.radius;
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
    if (!guide?.selectable || !guide.choices[index]) return;
    guide.selected = index;
    const graphic = guide.choices[index].graphic;
    const existing=(state.component.options.graphic||[]).filter(item=>item.id!=='dn-histogram-guide');
    state.chart.setOption({graphic:existing.concat(guide.visible?[graphic]:[])},{replaceMerge:['graphic']});
    state.component.$emit('plotDataset',{index});
    datasetMenu?.remove(); datasetMenu = null;
  }

  window.dnascapeHistogramMenu = event => {
    const root = event.event?.target?.closest?.('.dn-plot-navigation');
    const state = [...live].find(item => item.element === root);
    const guide = state?.component.options?.dnascapeHistogramGuides;
    if (!guide?.selectable || guide.choices.length < 2) return;
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

  function valueAt(state, axis, pixel, limits) {
    // Derive each scale independently. ECharts' Cartesian conversion can
    // choose a different axis pair when multiple boxplot axes share a grid.
    const rect = rectangle(state), horizontal = axis.dimension === 'x';
    const spec = state.chart.getModel().getComponent(axis.dimension + 'Axis', axis.index).option;
    if (rect.polar && Array.isArray(pixel)) {
      let fraction;
      if (horizontal) fraction = ((Math.atan2(pixel[0]-rect.cx,rect.cy-pixel[1])/(2*Math.PI))%1+1)%1;
      else {
        const radius = Math.hypot(pixel[0]-rect.cx,pixel[1]-rect.cy);
        let extent = [0,rect.radius];
        state.chart.getModel().eachComponent('polar', model => {
          const candidate = model.coordinateSystem.getRadiusAxis().getExtent().slice().sort((a,b)=>a-b);
          if (radius >= candidate[0] && radius <= candidate[1]) extent = candidate;
        });
        fraction = clamp((radius-extent[0])/(extent[1]-extent[0]),0,1);
      }
      if (axis.inverse) fraction=1-fraction;
      const [lower,upper]=limits||[spec.min,spec.max];
      return backward(forward(lower,axis)+fraction*(forward(upper,axis)-forward(lower,axis)),axis);
    }
    let fraction = (pixel - (horizontal ? rect.x : rect.y)) / (horizontal ? rect.width : rect.height);
    if (!horizontal) fraction = 1 - fraction;
    if (axis.inverse) fraction = 1 - fraction;
    const [lower, upper] = limits || [spec.min, spec.max];
    return backward(forward(lower, axis) + fraction * (forward(upper, axis) - forward(lower, axis)), axis);
  }

  function flush(state) {
    flushFrame(state);
    clearTimeout(state.timer);
    state.timer = null;
    if (!state.pending) return;
    const payload = state.pending;
    state.pending = null;
    if (payload.context !== metadata(state)?.context || !state.element.isConnected) return;
    payload.titleGaps = Object.fromEntries(Object.entries(metadata(state)?.axes || {})
      .filter(([, axis]) => axis.dimension === 'y')
      .map(([key, axis]) => [key, state.chart.getModel().getComponent('yAxis', axis.index).get('nameGap')]));
    state.component.$emit('plotViewport', payload);
  }

  function reserveTitleSpace(state) {
    if (metadata(state)?.polar) return;
    const grid = state.chart.getModel().getComponent('grid', 0), axes = [], margins = {};
    state.textMeasure ||= document.createElement('canvas').getContext('2d');
    for (const axis of Object.values(metadata(state)?.axes || {})) {
      if (axis.dimension !== 'y') continue;
      const model = state.chart.getModel().getComponent('yAxis', axis.index);
      if (!model.get('name') || model.get('nameLocation') !== 'middle') continue;
      const style = model.getModel('axisLabel');
      state.textMeasure.font = style.getFont();
      const width = Math.max(0, ...model.axis.getViewLabels().map(label =>
        state.textMeasure.measureText(label.formattedLabel).width));
      const current = Number(model.get('nameGap')) || 0;
      const required = Math.ceil(width + (Number(style.get('margin')) || 0) + 5);
      if (required <= current) continue;
      const side = model.axis.position === 'right' ? 'right' : 'left';
      margins[side] = (margins[side] ?? (Number(grid.get(side)) || 0)) + required - current;
      axes[axis.index] = {nameGap: required};
      const source = state.component.options.yAxis;
      (Array.isArray(source) ? source[axis.index] : source).nameGap = required;
    }
    if (!axes.length) return;
    const sourceGrid = state.component.options.grid;
    Object.assign(Array.isArray(sourceGrid) ? sourceGrid[0] : sourceGrid, margins);
    // Both updates finish in the same browser task. Keep the larger gutter
    // on zoom-out so shorter ticks cannot make the titles oscillate.
    state.chart.setOption({grid: margins, yAxis: Array.from({length: axes.length}, (_, i) => axes[i] || {})},
      {lazyUpdate: false, silent: true});
  }

  function axisPatch(state, updates, source) {
    const patch = {};
    for (const [key, values] of Object.entries(updates)) {
      const axis = metadata(state).axes[key], name = axis.dimension + 'Axis';
      if (!patch[name]) {
        const count = Math.max(...Object.values(metadata(state).axes).filter(item => item.dimension === axis.dimension).map(item => item.index)) + 1;
        const existing = source?.[name];
        patch[name] = Array.from({length: count}, (_, index) =>
          ({...(Array.isArray(existing) ? existing[index] : index === 0 ? existing : {})}));
      }
      const spec = patch[name][axis.index];
      Object.assign(spec, {min: values[0], max: values[1], interval: null});
      const sourceAxis = source ? spec : state.chart.getModel().getComponent(name, axis.index).option;
      const axisLine = sourceAxis.axisLine || {};
      spec.axisLine = {...axisLine, show:true, onZero:false,
        lineStyle:{width:1.2, ...axisLine.lineStyle, type:'solid', opacity:1}};
      for (const part of ['axisLabel', 'axisTick', 'splitLine']) spec[part] = {...sourceAxis[part], customValues: null};
      Object.assign(spec.axisLabel, {showMinLabel: true, showMaxLabel: true});
      if (axis.dimension === 'x' && sourceAxis.dnascapeBoxPositions) {
        const ticks = sourceAxis.dnascapeBoxPositions.filter(value => value >= values[0] && value <= values[1]);
        for (const part of ['axisLabel', 'axisTick', 'splitLine']) spec[part].customValues = ticks;
      } else if (axis.dimension === 'x' && !axis.log) {
        const ticks = horizontalTicks(...values);
        if (ticks.length) {
          for (const part of ['axisLabel', 'axisTick', 'splitLine']) spec[part].customValues = ticks;
          Object.assign(spec.splitLine, {showMinLine: true, showMaxLine: true});
        }
      }
    }
    if (metadata(state)?.polar) {
      const options=source||state.component.options;
      const current = ranges(state);
      const polarAxes = (name, fallback) => (options[name]||[]).map(axis=>{
        const values=updates[axis.dnascapeSourceAxis||fallback];
        return values?{...axis,min:values[0],max:values[1],interval:null}:axis;
      });
      patch.angleAxis=polarAxes('angleAxis','x');
      patch.radiusAxis=polarAxes('radiusAxis','y');
      patch.series=(options.series||[]).map(series=>{
        const key=options.angleAxis?.[series.polarIndex||0]?.dnascapeSourceAxis||'x';
        const window=updates[key]||current[key];
        if (series.dnascapeCircularMaskSourceData && window) {
          const ykey=options.radiusAxis?.[series.polarIndex||0]?.dnascapeSourceAxis||'y';
          const y=updates[ykey]||current[ykey];
          return {data:series.dnascapeCircularMaskSourceData.filter(p=>p[1]>window[0]&&p[0]<window[1])
            .map(p=>[Math.max(p[0],window[0]),Math.min(p[1],window[1]),y[1],y[0]])};
        }
        const data=series.dnascapeCircularSourceData;
        if(!data||!window)return {};
        if(options.dnascapePlotType==='Histogram'&&series.type==='custom'&&!series.silent)
          return {data:data.filter(p=>p[1]>window[0]&&p[0]<window[1]).map(p=>[Math.max(p[0],window[0]),Math.min(p[1],window[1]),...p.slice(2)])};
        return {data:data.filter(p=>p[0]>=window[0]&&p[0]<=window[1]&&(p.length!==6||p[3]<=window[1]))};
      });
    }
    return patch;
  }

  function queueViewport(state) {
    state.pending = {context: metadata(state).context, token: metadata(state).token,
      sequence: state.sequence, ranges: state.latestRanges};
    // A fixed trailing window also sends updates during a long drag. Do not
    // restart it for every mouse event and starve the detail prefetch.
    if (!state.timer) state.timer = setTimeout(() => flush(state), 120);
  }

  function apply(state, next, notify = true) {
    const current = ranges(state);
    const updates = Object.fromEntries(Object.entries(next).filter(([key, values]) => changed(current[key], values)));
    if (!Object.keys(updates).length) return;
    const patch = axisPatch(state, updates);
    state.chart.setOption(patch, {lazyUpdate: false, silent: true});
    reserveTitleSpace(state);
    state.sequence = ++sequence;
    state.latestRanges = {...current, ...updates};
    if (notify) {
      queueViewport(state);
    }
  }

  function flushFrame(state) {
    if (state.frame === undefined) return;
    cancelAnimationFrame(state.frame);
    state.frame = undefined;
    const next = state.next;
    state.next = null;
    apply(state, next);
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
    flush(state);
    const rect = rectangle(state), start = point(state, event), initial = ranges(state);
    if (!inside(rect, start) || inLegend(state, start) || inside(guideBox(state),start) || !Object.keys(initial).length) return;
    const mode = leftControl && event.ctrlKey ? 'select' : 'pan';
    if (mode === 'select' && !Object.keys(initial).some(key => key.endsWith('x'))) return;
    event.preventDefault();
    state.suppressClickUntil = 0;
    state.chart.dispatchAction({type: 'hideTip'});
    state.chart.dispatchAction({type: 'updateAxisPointer', currTrigger: 'leave'});
    if (mode === 'select') event.stopImmediatePropagation();
    drag = {state, mode, start, end: start, rect, initial, context: metadata(state).context, box:!!metadata(state).heatmap};
    if (mode === 'select') {
      const overlay = document.createElement('div');
      overlay.className = 'dn-plot-range-selection';
      overlay.setAttribute('aria-hidden', 'true');
      overlay.style.cssText = `left:${start[0]}px;top:${rect.y}px;width:0;height:${rect.height}px;`;
      if (drag.box) overlay.style.cssText=`left:${start[0]}px;top:${start[1]}px;width:0;height:0;`;
      if (rect.polar) overlay.style.cssText=`left:${rect.x}px;top:${rect.y}px;width:${rect.width}px;height:${rect.height}px;border-radius:50%;background:transparent;`;
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
    if (Math.hypot(drag.end[0] - start[0], drag.end[1] - start[1]) >= 3) drag.moved = true;
    if (drag.mode === 'select') {
      const end = clamp(drag.end[0], rect.x, rect.x + rect.width);
      drag.overlay.style.left = Math.min(start[0], end) + 'px';
      drag.overlay.style.width = Math.abs(end - start[0]) + 'px';
      if (drag.box) {
        const y=clamp(drag.end[1],rect.y,rect.y+rect.height);
        drag.overlay.style.top=Math.min(start[1],y)+'px';
        drag.overlay.style.height=Math.abs(y-start[1])+'px';
      }
      if (rect.polar) {
        const angle=p=>((Math.atan2(p[0]-rect.cx,rect.cy-p[1])*180/Math.PI)%360+360)%360;
        const a=Math.min(angle(start),angle(drag.end)),b=Math.max(angle(start),angle(drag.end));
        Object.assign(drag.overlay.style,{left:rect.x+'px',width:rect.width+'px',
          background:`conic-gradient(transparent 0deg ${a}deg,rgba(176,52,18,.15) ${a}deg ${b}deg,transparent ${b}deg 360deg)`});
      }
    } else {
      if (!drag.moved) return;
      const next = {};
      for (const [key, values] of Object.entries(initial)) {
        const axis = metadata(state).axes[key], horizontal = axis.dimension === 'x';
        const delta = drag.end[horizontal ? 0 : 1] - start[horizontal ? 0 : 1];
        const lower = forward(values[0], axis), upper = forward(values[1], axis);
        const polarDelta = rect.polar ? (horizontal
          ? Math.atan2(Math.sin(Math.atan2(drag.end[0]-rect.cx,rect.cy-drag.end[1])-Math.atan2(start[0]-rect.cx,rect.cy-start[1])),
                       Math.cos(Math.atan2(drag.end[0]-rect.cx,rect.cy-drag.end[1])-Math.atan2(start[0]-rect.cx,rect.cy-start[1])))/(2*Math.PI)
          : -(Math.hypot(drag.end[0]-rect.cx,drag.end[1]-rect.cy)-Math.hypot(start[0]-rect.cx,start[1]-rect.cy))/rect.radius) : null;
        const shift = (polarDelta ?? delta / (horizontal ? rect.width : rect.height)) * (upper - lower) *
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
    if (drag.mode === 'pan' && !drag.moved && Math.hypot(pointAtUp[0]-drag.start[0],pointAtUp[1]-drag.start[1]) < 3) {
      const state=drag.state; drag=null; flushFrame(state); cursor(state,pointAtUp); return;
    }
    move(event);
    if (!drag) return;
    const {state, initial, mode, start, end, rect} = drag;
    state.suppressClickUntil = Date.now() + 250;
    suppressNavigationClickUntil = state.suppressClickUntil;
    flushFrame(state);
    if (mode === 'select' && (rect.polar ? Math.hypot(end[0]-start[0],end[1]-start[1]) : Math.abs(end[0]-start[0])) >= 3
        && (!drag.box || Math.abs(end[1]-start[1]) >= 3)) {
      const next = {};
      for (const [key, axis] of Object.entries(metadata(state).axes)) {
        if (axis.dimension !== 'x' && !drag.box) continue;
        const horizontal=axis.dimension==='x';
        const pixels=rect.polar?[start,end]:[start[horizontal?0:1],clamp(end[horizontal?0:1],horizontal?rect.x:rect.y,(horizontal?rect.x+rect.width:rect.y+rect.height))];
        const values = pixels.map(pixel => valueAt(state, axis, pixel)).sort((a, b) => a - b);
        if (values.every(Number.isFinite)) next[key] = bounded(forward(values[0], axis), forward(values[1], axis), axis);
      }
      apply(state, next);
    } else if (mode === 'pan') {
      const current = ranges(state);
      const moved = Object.fromEntries(Object.entries(current).filter(([key, values]) => changed(initial[key], values)));
      if (Object.keys(moved).length) queueViewport(state);
    }
    drag.overlay?.remove();
    drag = null;
    flush(state);
    cursor(state, end);
  }

  function cancel() {
    if (!drag) return;
    const {state, initial} = drag;
    if (drag.moved) suppressNavigationClickUntil = state.suppressClickUntil = Date.now() + 250;
    if (state.frame !== undefined) cancelAnimationFrame(state.frame);
    state.frame = undefined;
    state.next = null;
    if (state.element.isConnected && !state.chart.isDisposed() && drag.context === metadata(state)?.context) {
      apply(state, initial);
      flush(state);
    }
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
    for (const [key, values] of Object.entries(state.next || ranges(state))) {
      const axis = axes[key], lower = forward(values[0], axis), upper = forward(values[1], axis);
      const value = valueAt(state, axis, rect.polar ? position : position[axis.dimension === 'x' ? 0 : 1], values);
      const anchor = clamp(forward(value, axis), lower, upper);
      if (!Number.isFinite(anchor)) continue;
      next[key] = bounded(anchor - (anchor - lower) * factor, anchor + (upper - anchor) * factor, axis);
    }
    paint(state, next);
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
      const identity = value => JSON.stringify([value?.context, value?.token]);
      state.identity = identity(metadata(state));
      state.sequence = Number(metadata(state)?.sequence) || 0;
      sequence = Math.max(sequence, state.sequence);
      const setOption = chart.setOption.bind(chart);
      chart.setOption = (options, ...args) => {
        const incoming = options.dnascapeNavigation;
        if (incoming) {
          const acknowledged = Number(incoming.sequence) || 0;
          if (identity(incoming) !== state.identity) {
            if (drag?.state === state) cancel();
            clearTimeout(state.timer);
            if (state.frame !== undefined) cancelAnimationFrame(state.frame);
            state.frame = undefined;
            state.timer = state.pending = state.next = state.latestRanges = null;
            state.identity = identity(incoming);
            state.sequence = acknowledged;
          } else if ((state.latestRanges || state.next) &&
                     (acknowledged < state.sequence || state.next || drag?.state === state)) {
            // A server response can arrive after another gesture has begun.
            // Install its detail and styles in the current viewport, without
            // ever painting the older axis ranges, even for a single frame.
            options = {...options, ...axisPatch(state, state.next || state.latestRanges, options)};
          } else {
            state.sequence = acknowledged;
            state.latestRanges = null;
          }
          sequence = Math.max(sequence, acknowledged);
        }
        return setOption(options, ...args);
      };
      const onClick = event => {
        if (state.suppressClickUntil > Date.now()) return;
        if (component.options?.dnascapeHistogramGuides?.selectable && event.componentType === 'series') {
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
    if (pressed && event.code !== 'Escape') suppressNavigationClickUntil = 0;
    if (event.code === 'ControlLeft') leftControl = pressed;
    if (event.code === 'ControlLeft' && !pressed) for (const state of live) flush(state);
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
    suppressNavigationClickUntil = 0;
    for (const state of live) flush(state);
    if (datasetMenu && !datasetMenu.contains(event.target)) { datasetMenu.remove(); datasetMenu=null; }
  }, true);
  document.addEventListener('click', event => {
    if (Date.now() < suppressNavigationClickUntil) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);
  new MutationObserver(schedule).observe(document.body, {childList: true, subtree: true});
  schedule();
})();
