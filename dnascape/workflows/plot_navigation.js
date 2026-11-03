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
    const period=ceiling-floor;
    const start = axis.circular && span<period*(1-1e-9)
      ? floor+((lower-floor)%period+period)%period : clamp(lower, floor, ceiling-span);
    return [backward(start, axis), backward(start + span, axis)];
  }

  function circularParts(axis, window) {
    const floor=forward(axis.min,axis),ceiling=forward(axis.max,axis),period=ceiling-floor;
    const span=forward(window[1],axis)-forward(window[0],axis);
    if(span>=period*(1-1e-9))return [[axis.min,axis.max]];
    const first=floor+((forward(window[0],axis)-floor)%period+period)%period,last=first+span;
    return last<=ceiling+period*1e-12?[[backward(first,axis),backward(Math.min(last,ceiling),axis)]]:
      [[backward(first,axis),axis.max],[axis.min,backward(floor+last-ceiling,axis)]];
  }

  function shortArc(rect, start, end) {
    const angle=p=>((Math.atan2(p[0]-rect.cx,rect.cy-p[1])/(2*Math.PI))%1+1)%1;
    const first=angle(start),last=angle(end),span=(last-first+1)%1;
    return span<=.5?[first,first+span]:[last,last+1-span];
  }

  function metadata(state) {
    return state.component.options?.dnascapeNavigation;
  }

  window.dnascapeClipCircularFrame = chart => {
    if(chart.isDisposed())return;
    const globalModel=chart.getModel();
    if(!globalModel)return;
    let prototype;
    globalModel.eachComponent('graphic',model=>{
      chart.getViewOfComponentModel(model)?.group.traverse(element=>{
        if(element.type==='rect'&&element.shape.dnascapeCircularFrame)prototype=element;
      });
    });
    if(!prototype)return;
    const width=chart.getWidth(),height=Math.max(1,chart.getHeight()-28);
    const clip=group=>{
      if(!group)return;
      const existing=group.getClipPath();
      let frame=group.__dnascapeCircularFrame;
      if(existing!==frame||!frame){
        frame=new prototype.constructor({shape:{x:0,y:0,width,height}});
        if(existing)frame.setClipPath(existing);
        group.setClipPath(frame);group.__dnascapeCircularFrame=frame;
      }else if(frame.shape.width!==width||frame.shape.height!==height){
        frame.setShape({width,height});
      }
    };
    globalModel.eachSeries(model=>{
      if(model.get('coordinateSystem')==='polar')clip(chart.getViewOfSeriesModel(model)?.group);
    });
    for(const name of ['angleAxis','radiusAxis'])globalModel.eachComponent(name,model=>clip(chart.getViewOfComponentModel(model)?.group));
    return true;
  };

  function installCircularFrame(chart) {
    if(chart.__dnascapeFrameInstalled)return;
    chart.__dnascapeFrameInstalled=true;
    const setOption=chart.setOption.bind(chart),resize=chart.resize.bind(chart),capture=chart.getDataURL.bind(chart);
    chart.setOption=(options,...args)=>{
      // NiceGUI only replaces options when the series count changes. A view
      // switch can keep that count while changing coordinate systems; merging
      // then retains polar axes, hidden Cartesian axes and polar series styles.
      // Only full specifications carry the plot type: gesture patches must merge.
      if(options.dnascapePlotType&&!!chart.getOption()?.dnascapeCircular!==!!options.dnascapeCircular){
        args[0]=typeof args[0]==='object'?{...args[0],notMerge:true}:true;
      }
      const result=setOption(options,...args);
      window.dnascapeClipCircularFrame(chart);
      return result;
    };
    chart.resize=(...args)=>{const result=resize(...args);window.dnascapeClipCircularFrame(chart);return result;};
    chart.getDataURL=(...args)=>{
      if(window.dnascapeClipCircularFrame(chart))chart.getZr().refreshImmediately();
      return capture(...args);
    };
    chart.on('rendered',()=>window.dnascapeClipCircularFrame(chart));
    window.dnascapeClipCircularFrame(chart);
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
      const [lower,upper]=horizontal?[axis.min,axis.max]:(limits||[spec.min,spec.max]);
      let value=forward(lower,axis)+fraction*(forward(upper,axis)-forward(lower,axis));
      if(horizontal&&axis.circular){
        const window=limits||[spec.min,spec.max],period=forward(upper,axis)-forward(lower,axis);
        const middle=(forward(window[0],axis)+forward(window[1],axis))/2;
        value+=Math.round((middle-value)/period)*period;
      }
      return backward(value,axis);
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

  function circularCamera(options, window) {
    const base=options.dnascapeCircularBasePolar;
    const axis=options.angleAxis?.[0];
    if(!base||!axis||!window)return options.polar;
    const convert=value=>axis.type==='log'?Math.log(value):value;
    const lo=convert(axis.min),hi=convert(axis.max),a=convert(window[0]),b=convert(window[1]);
    if(Math.abs((b-a)/(hi-lo)-1)<1e-9)return base;
    let first=(a-lo)/(hi-lo),last=(b-lo)/(hi-lo);
    if(axis.inverse)[first,last]=[1-last,1-first];
    const samples=[first,last];
    for(let tick=Math.ceil(first*4);tick<=Math.floor(last*4);tick++)samples.push(tick/4);
    const radii=base.flatMap(p=>p.radius.map(value=>parseFloat(value)/100));
    const points=[Math.min(...radii),Math.max(...radii)].flatMap(radius=>samples.map(f=>[radius*Math.sin(2*Math.PI*f),-radius*Math.cos(2*Math.PI*f)]));
    const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
    const scale=Math.min(200,Math.max(1,Math.min(1.72/Math.max(xmax-xmin,1e-9),1.6/Math.max(ymax-ymin,1e-9))));
    const center=[50-25*(xmin+xmax)*scale,50-25*(ymin+ymax)*scale].map(value=>value+'%');
    return base.map(p=>({...p,center,radius:p.radius.map(value=>parseFloat(value)*scale+'%')}));
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
        const key=axis.dnascapeSourceAxis||fallback;
        const values=updates[key]||current[key];
        return values?{...axis,min:values[0],max:values[1],interval:null}:axis;
      });
      // Keep the angular coordinate system fixed. Zoom changes only the
      // camera and the sampled window, never the arc's position in the circle.
      patch.angleAxis=(options.angleAxis||[]).map(axis=>{
        const key=axis.dnascapeSourceAxis||'x',window=updates[key]||current[key];
        const domain=metadata(state).axes[key];
        const ticks=window&&axis.type!=='log'?[...new Set(circularParts(domain,window).flatMap(part=>horizontalTicks(...part)))].sort((a,b)=>a-b):null;
        const result={...axis,axisPointer:{...axis.axisPointer,show:false}};
        if(ticks)for(const part of ['axisLabel','axisTick','splitLine'])result[part]={...axis[part],customValues:ticks};
        return result;
      });
      patch.radiusAxis=polarAxes('radiusAxis','y');
      patch.polar=circularCamera(options,updates.x||current.x);
      for(const name of ['xAxis','yAxis'])if(patch[name])for(const axis of patch[name])axis.axisPointer={show:false,label:{show:false}};
      patch.series=(options.series||[]).map(series=>{
        // A native gesture only changes data. A delayed full replacement also
        // needs its series definitions, or masks and uncertainty bands vanish.
        const style=source?series:{};
        if (series.dnascapeCircularMaskSourceData) {
          const ykey=options.radiusAxis?.[series.polarIndex||0]?.dnascapeSourceAxis||'y';
          const y=updates[ykey]||current[ykey];
          return {...style,data:series.dnascapeCircularMaskSourceData.map(p=>[p[0],p[1],y[1],y[0]])};
        }
        const data=series.dnascapeCircularSourceData;
        // Retain the coarse surrounding profile and prefetched detail. The
        // canvas frame clips the magnified circle, not the selected endpoints.
        return data?{...style,data}:{...style};
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

  function clearCircularHover(state) {
    if(state.hoverFrame!==undefined)cancelAnimationFrame(state.hoverFrame);
    state.hoverFrame=undefined;
    if(state.circularTooltip)state.circularTooltip.style.display='none';
    if(state.circularPosition)state.circularPosition.style.display='none';
    if(state.circularGuide)state.circularGuide.style.display='none';
  }

  function circularHover(state, position) {
    if(!metadata(state)?.polar)return;
    const rect=rectangle(state),options=state.component.options;
    if(drag||!options.tooltip?.show||!inside(rect,position)||inLegend(state,position))return clearCircularHover(state);
    const width=state.chart.getWidth(),height=state.chart.getHeight();
    const angle=Math.atan2(position[0]-rect.cx,rect.cy-position[1]);
    const fraction=((angle/(2*Math.PI))%1+1)%1;
    const points=[],rows=[],anchors=[];
    state.chart.getModel().eachSeries(series=>{
      if(series.get('silent')||series.get('coordinateSystem')!=='polar'||!['line','scatter','custom'].includes(series.subType))return;
      const name=series.get('name');let selected=true;
      state.chart.getModel().eachComponent('legend',legend=>{if(!legend.isSelected(name))selected=false;});
      if(!selected)return;
      const polar=series.coordinateSystem,axis=polar.getAngleAxis(),spec=axis.model.option;
      const key=spec.dnascapeSourceAxis||'x',scale=metadata(state).axes[key];
      if(!scale)return;
      const f=spec.inverse?1-fraction:fraction;
      const target=backward(forward(scale.min,scale)+f*(forward(scale.max,scale)-forward(scale.min,scale)),scale);
      const data=series.getData(),count=data.count();if(!count)return;
      const valueAt=index=>{const value=data.getRawDataItem(index);return Array.isArray(value)?value:value?.value;};
      let lo=0,hi=count;
      while(lo<hi){const mid=(lo+hi)>>1; if(valueAt(mid)?.[0]<target)lo=mid+1;else hi=mid;}
      let index=clamp(lo,0,count-1);
      const histogram=options.dnascapePlotType==='Histogram'&&series.subType==='custom';
      if(index>0&&(histogram?valueAt(index-1)[1]>=target:Math.abs(valueAt(index-1)[0]-target)<Math.abs(valueAt(index)[0]-target)))index--;
      const value=valueAt(index);if(!value)return;
      const x=histogram?clamp(target,value[0],value[1]):value[0],y=histogram?value[2]:value[1];
      if(!Number.isFinite(x)||!Number.isFinite(y))return;
      const point=polar.dataToPoint([y,x]);
      const params=series.getDataParams(index),color=String(params.color||series.get('lineStyle.color')||'#b03412');
      params.marker='<span style="display:inline-block;margin-right:5px;border-radius:50%;width:10px;height:10px;background-color:'+color+'"></span>';
      if(point.every(Number.isFinite)&&point[0]>=0&&point[0]<=width&&point[1]>=0&&point[1]<=height){
        rows.push(params);
        anchors.push({point,value:x,axis:spec});
        points.push({point,color});
      }
    });
    if(!anchors.length)return clearCircularHover(state);
    const nearest=anchors.reduce((a,b)=>Math.hypot(a.point[0]-position[0],a.point[1]-position[1])<=Math.hypot(b.point[0]-position[0],b.point[1]-position[1])?a:b);
    const radius=Math.hypot(nearest.point[0]-rect.cx,nearest.point[1]-rect.cy);
    const guideRadius=Math.max(...anchors.map(entry=>Math.hypot(entry.point[0]-rect.cx,entry.point[1]-rect.cy)));
    const hoverHeight=Math.max(1,height-28); // Keep the fixed position title unobscured.
    const unit=radius>1e-9?[(nearest.point[0]-rect.cx)/radius,(nearest.point[1]-rect.cy)/radius]:[Math.sin(angle),-Math.cos(angle)];
    const axis=nearest.axis;
    const value=nearest.value;
    const formatter=axis.axisPointer?.label?.formatter;
    const label=typeof formatter==='function'?String(formatter({value})):String(Number(value.toPrecision(6)));
    const accent=getComputedStyle(document.body).getPropertyValue('--dn-accent').trim()||axis.axisPointer?.label?.backgroundColor||'#b03412';
    // Hover is a screen-space overlay. Updating ECharts' graphic component on
    // every mouse move can drop its children and relayout the fixed axis title.
    const svgNS='http://www.w3.org/2000/svg';
    const guide=state.circularGuide||(state.circularGuide=document.createElementNS(svgNS,'svg'));
    if(!guide.parentElement){
      if(getComputedStyle(state.element).position==='static')state.element.style.position='relative';
      guide.classList.add('dn-circular-guide');guide.setAttribute('aria-hidden','true');
      guide.style.cssText='position:absolute;left:0;top:0;z-index:1000;pointer-events:none;overflow:hidden;';
      const line=document.createElementNS(svgNS,'line');
      line.setAttribute('stroke','#8e8781');line.setAttribute('stroke-width','1');line.setAttribute('stroke-dasharray','4 4');
      guide.appendChild(line);state.element.appendChild(guide);
    }
    guide.setAttribute('viewBox',`0 0 ${width} ${hoverHeight}`);guide.style.width=width+'px';guide.style.height=hoverHeight+'px';guide.style.display='block';
    for(const [key,value] of Object.entries({x1:rect.cx,y1:rect.cy,x2:rect.cx+unit[0]*(guideRadius+10),y2:rect.cy+unit[1]*(guideRadius+10)}))
      guide.firstChild.setAttribute(key,String(value));
    while(guide.children.length>points.length+1)guide.lastChild.remove();
    for(const [index,entry] of points.entries()){
      let dot=guide.children[index+1];
      if(!dot){dot=document.createElementNS(svgNS,'circle');dot.setAttribute('r','3.5');dot.setAttribute('stroke','#fff');dot.setAttribute('stroke-width','1');guide.appendChild(dot);}
      dot.setAttribute('cx',String(entry.point[0]));dot.setAttribute('cy',String(entry.point[1]));dot.setAttribute('fill',entry.color);
    }
    const readout=state.circularPosition||(state.circularPosition=document.createElement('div'));
    if(!readout.parentElement){
      readout.className='dn-circular-position';
      readout.style.cssText='position:absolute;z-index:1002;pointer-events:none;color:#fff;font:normal 11px sans-serif;line-height:15px;padding:3px 6px;border-radius:3px;white-space:nowrap;';
      state.element.appendChild(readout);
    }
    readout.textContent=label;readout.style.backgroundColor=accent;readout.style.display='block';
    const padding=Math.max(18,Math.abs(unit[0])*readout.offsetWidth/2+Math.abs(unit[1])*readout.offsetHeight/2+10);
    readout.style.left=Math.round(clamp(nearest.point[0]+unit[0]*padding-readout.offsetWidth/2,2,width-readout.offsetWidth-2))+'px';
    readout.style.top=Math.round(clamp(nearest.point[1]+unit[1]*padding-readout.offsetHeight/2,2,Math.max(2,hoverHeight-readout.offsetHeight-2)))+'px';
    const tooltip=state.circularTooltip||(state.circularTooltip=document.createElement('div'));
    if(!tooltip.parentElement){
      tooltip.className='dn-circular-tooltip';
      tooltip.style.cssText='position:absolute;z-index:1001;pointer-events:none;background:#fff;border:1px solid #ddd7d1;border-radius:4px;padding:8px 10px;color:#625c57;font:14px sans-serif;box-shadow:0 2px 8px #0002;';
      state.element.appendChild(tooltip);
    }
    const format=state.chart.getModel().getComponent('tooltip')?.get('formatter');
    tooltip.innerHTML=rows.length?(typeof format==='function'?format(rows):rows.map(p=>p.marker+'<b>'+p.value[1]+'</b>').join('<br>')):'';
    tooltip.style.display=rows.length?'block':'none';
    const tw=tooltip.offsetWidth,th=tooltip.offsetHeight,rx=parseFloat(readout.style.left),ry=parseFloat(readout.style.top);
    const candidates=[[12,12],[12,-th-12],[-tw-12,12],[-tw-12,-th-12]].map(([dx,dy])=>{
      const x=clamp(nearest.point[0]+dx,2,Math.max(2,width-tw-2)),y=clamp(nearest.point[1]+dy,2,Math.max(2,hoverHeight-th-2));
      const overlap=Math.max(0,Math.min(x+tw,rx+readout.offsetWidth)-Math.max(x,rx))*
        Math.max(0,Math.min(y+th,ry+readout.offsetHeight)-Math.max(y,ry));
      return {x,y,overlap};
    });
    const placement=candidates.reduce((a,b)=>a.overlap<=b.overlap?a:b);
    tooltip.style.left=Math.round(placement.x)+'px';tooltip.style.top=Math.round(placement.y)+'px';
  }

  function queueCircularHover(state, position) {
    state.hoverPoint=position;
    if(state.hoverFrame!==undefined)return;
    state.hoverFrame=requestAnimationFrame(()=>{state.hoverFrame=undefined;circularHover(state,state.hoverPoint);});
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
    clearCircularHover(state);
    state.suppressClickUntil = 0;
    state.chart.dispatchAction({type: 'hideTip'});
    state.chart.dispatchAction({type: 'updateAxisPointer', currTrigger: 'leave'});
    if (mode === 'select') event.stopImmediatePropagation();
    drag = {state, mode, start, end: start, rect, initial, context: metadata(state).context, box:!!metadata(state).heatmap};
    if (mode === 'select') {
      if(getComputedStyle(state.element).position==='static')state.element.style.position='relative';
      const overlay = document.createElement('div');
      overlay.className = 'dn-plot-range-selection';
      overlay.setAttribute('aria-hidden', 'true');
      overlay.style.cssText = `left:${start[0]}px;top:${rect.y}px;width:0;height:${rect.height}px;`;
      if (drag.box) overlay.style.cssText=`left:${start[0]}px;top:${start[1]}px;width:0;height:0;`;
      if (rect.polar) overlay.style.cssText=`left:0;top:0;width:${state.chart.getWidth()}px;height:${state.chart.getHeight()}px;border:0;border-radius:0;background:transparent;clip-path:circle(${rect.radius}px at ${rect.cx}px ${rect.cy}px);`;
      Object.assign(overlay.style,{position:'absolute',pointerEvents:'none',zIndex:'1003'});
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
        const [a,b]=shortArc(rect,start,drag.end);
        Object.assign(drag.overlay.style,{left:'0px',top:'0px',width:state.chart.getWidth()+'px',height:state.chart.getHeight()+'px',
          background:`conic-gradient(from ${a*360}deg at ${rect.cx}px ${rect.cy}px,color-mix(in srgb,var(--dn-accent,#b03412) 15%,transparent) 0deg ${(b-a)*360}deg,transparent ${(b-a)*360}deg 360deg)`});
      }
    } else {
      if (!drag.moved) return;
      const next = {};
      for (const [key, values] of Object.entries(initial)) {
        const axis = metadata(state).axes[key], horizontal = axis.dimension === 'x';
        if(rect.polar&&!horizontal)continue;
        const delta = drag.end[horizontal ? 0 : 1] - start[horizontal ? 0 : 1];
        const lower = forward(values[0], axis), upper = forward(values[1], axis);
        const polarDelta = rect.polar ? (horizontal
          ? Math.atan2(Math.sin(Math.atan2(drag.end[0]-rect.cx,rect.cy-drag.end[1])-Math.atan2(start[0]-rect.cx,rect.cy-start[1])),
                       Math.cos(Math.atan2(drag.end[0]-rect.cx,rect.cy-drag.end[1])-Math.atan2(start[0]-rect.cx,rect.cy-start[1])))/(2*Math.PI)
          : -(Math.hypot(drag.end[0]-rect.cx,drag.end[1]-rect.cy)-Math.hypot(start[0]-rect.cx,start[1]-rect.cy))/rect.radius) : null;
        const shift = (polarDelta ?? delta / (horizontal ? rect.width : rect.height)) *
          (rect.polar?forward(axis.max,axis)-forward(axis.min,axis):upper-lower) *
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
        if(rect.polar){
          let [first,last]=shortArc(rect,start,end);
          if((last-first)*2*Math.PI*rect.radius<3)continue;
          if(axis.inverse)[first,last]=[1-last,1-first];
          const floor=forward(axis.min,axis),span=forward(axis.max,axis)-floor;
          next[key]=bounded(floor+first*span,floor+last*span,axis);
          continue;
        }
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
    if (!inside(rect, position) || inLegend(state, position) || inside(guideBox(state),position)) return;
    const axes = metadata(state)?.axes;
    if (!axes || !Object.keys(axes).length) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1);
    const factor = Math.exp(clamp(pixels * 0.002, -1, 1)), next = {};
    for (const [key, values] of Object.entries(state.next || ranges(state))) {
      const axis = axes[key], lower = forward(values[0], axis), upper = forward(values[1], axis);
      if(rect.polar&&axis.dimension!=='x')continue;
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
    document.querySelectorAll('.nicegui-echart').forEach(element => {
      const component = getElement(Number(element.id.slice(1))), chart = component?.chart;
      if (!chart) { pending = true; return; }
      installCircularFrame(chart);
      if(!element.classList.contains('dn-plot-navigation'))return;
      if (attached.has(chart)) return;
      const state = {element, component, chart};
      const identity = value => JSON.stringify([value?.context, value?.token]);
      state.identity = identity(metadata(state));
      state.sequence = Number(metadata(state)?.sequence) || 0;
      sequence = Math.max(sequence, state.sequence);
      const setOption = chart.setOption.bind(chart);
      chart.setOption = (options, ...args) => {
        const incoming = options.dnascapeNavigation;
        if(incoming)clearCircularHover(state);
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
        const result=setOption(options, ...args);
        if(incoming&&state.lastPoint&&metadata(state)?.polar)queueCircularHover(state,state.lastPoint);
        return result;
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
      const onMove = event => { if (!drag) { state.lastPoint = point(state, event); cursor(state, state.lastPoint);
        if(metadata(state)?.polar)queueCircularHover(state,state.lastPoint); } };
      const onLeave=()=>{state.lastPoint=null;clearCircularHover(state);};
      element.addEventListener('mousedown', onDown, true);
      element.addEventListener('wheel', onWheel, {capture: true, passive: false});
      element.addEventListener('mousemove', onMove);
      element.addEventListener('mouseleave', onLeave);
      state.dispose = () => {
        clearTimeout(state.timer);
        clearCircularHover(state);
        state.circularTooltip?.remove();
        state.circularPosition?.remove();
        state.circularGuide?.remove();
        chart.off('click',onClick);
        if (state.frame !== undefined) cancelAnimationFrame(state.frame);
        element.removeEventListener('mousedown', onDown, true);
        element.removeEventListener('wheel', onWheel, true);
        element.removeEventListener('mousemove', onMove);
        element.removeEventListener('mouseleave', onLeave);
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
  window.addEventListener('resize', ()=>{cancel();for(const state of live)clearCircularHover(state);});
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
