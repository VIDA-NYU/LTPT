import {fabric} from "fabric";
import {schemeTableau10} from "d3-scale-chromatic"
import {Streamlit} from 'streamlit-component-lib'
function isAroundPoint(pointer, coords) {

    let dist = (pointer.x - coords[0]) * (pointer.x - coords[0]) + (pointer.y - coords[1]) * (pointer.y - coords[1]);
    dist = Math.sqrt(dist)
    if (dist < 10) {
        return {isAround: true, dist}
    } else {
        return {isAround: false, dist}
    }
}

function isAroundLine(pointer, coords){
    let v1 = [pointer.x - coords[0], pointer.y - coords[1]];
    let v2 = [coords[2] - coords[0], coords[3] - coords[1]]
    let d = Math.sqrt(Math.pow(v2[0], 2) + Math.pow(v2[1], 2) );
    let t = (v1[0] * v2[0] + v1[1] * v2[1]) / Math.sqrt(Math.pow(v2[0], 2) + Math.pow(v2[1], 2));
    let lPower = Math.pow(v1[0], 2) + Math.pow(v1[1], 2);
    let dist = Math.sqrt(lPower - Math.pow(t, 2));
    let nearPoint = {
        x: coords[0] + v2[0] * t / d,
        y: coords[1] + v2[1] * t /d
    }

    if (dist > 10) {
        return {isAround: false, dist, lineNearPoint: null}
    } else {
        return {isAround: true, dist, lineNearPoint: nearPoint}
    }
}


var lines = [];
function onDrawLine(canvas, fixedPoints, styles, addAMetric) {
    let line = null;
    let isDown = false;
    let isMetricDrawing = false;
    let startJoint = -1;
    let metricStartLine = -1;
    function aroundWhichPoint(pointer) {
        let selectedIdx = -1;
        let minDist = -1;
        for (let i in fixedPoints) {
            let point = fixedPoints[i];
            let {isAround, dist} = isAroundPoint(pointer, point.coords);
            if (isAround) {
                if (minDist < 0 | minDist > dist) {
                    dist = minDist;
                    selectedIdx = i;
                }
            }
        }
        if (selectedIdx < 0) {
            return selectedIdx;
        }
        return selectedIdx;
    }

    function aroundWhichLine(pointer){
         let selectedIdx = -1;
        let minDist = -1;
        let nearPoint = null;
        for (let i in lines) {
            let lineData = lines[i];
            let src = lineData.src;
            let dest = lineData.dest;
            let srcPoint = fixedPoints.filter(d=>d.name===src)[0];
            let destPoint = fixedPoints.filter(d=>d.name===dest)[0];
            let coords = [
                srcPoint.coords[0], srcPoint.coords[1],
                destPoint.coords[0], destPoint.coords[1]
            ];
            let point = fixedPoints[i];
            let {isAround, dist,  lineNearPoint} = isAroundLine(pointer, coords);
            if (isAround) {
                if (minDist < 0 | minDist > dist) {
                    dist = minDist;
                    selectedIdx = i;
                    nearPoint = lineNearPoint

                }
            }
        }
        if (selectedIdx < 0) {
            return {selectedIdx, nearPoint: null};
        }
        return {lineIdx: selectedIdx, nearPoint};
    }

    function finishLine(pointer) {
        let pointIdx = aroundWhichPoint(pointer);
        if (pointIdx >= 0) {
            line.setCoords();
            line.set({
                x2: fixedPoints[pointIdx].coords[0],
                y2: fixedPoints[pointIdx].coords[1]
            })
            isDown = false;
            let lineData = {
                src: fixedPoints[pointIdx].name,
                dest: fixedPoints[startJoint].name,
                color: "grey"
            }
            lines.push(lineData);
            line = null;

        }
    }

    function startLine(pointer) {
        let points = [pointer.x, pointer.y, pointer.x, pointer.y];
        let pointIdx = aroundWhichPoint(pointer);
        if (pointIdx >= 0) {
            line = new fabric.Line(points, {
                stroke: "grey",
                hasControls: false,
                hasBorders: false,
                lockMovementX: false,
                lockMovementY: false,
                hoverCursor: 'default',
                selectable: false,
            });
            line.set({
                x1: fixedPoints[pointIdx].coords[0],
                y1: fixedPoints[pointIdx].coords[1]
            })
            canvas.add(line);
            isDown = true;
            startJoint = pointIdx;
            return true;
        }else{
            return false;
        }

    }
    function finishDrawingMetric(pointer) {
        let {lineIdx, nearPoint} = aroundWhichLine(pointer);
        if (lineIdx >= 0) {
            line.setCoords();
            line.set({
                x2: nearPoint.x,
                y2: nearPoint.y
            })
            isMetricDrawing = false;
            line = null;
            let metric = {

            }
            if(lineIdx === metricStartLine){
                metric['type'] = "Distance";
                metric['lines'] = [lineIdx].map(d=>lines[d])
            }else{
                metric['type'] = "Angle";
                metric['lines'] = [lineIdx, metricStartLine].map(d=>lines[d])
            }
            addAMetric({
                type: metric['type'],
                name: "default name",
                lines: metric['lines']

            })

        }
    }
    function startDrawingMetric(pointer){
        let points = [pointer.x, pointer.y, pointer.x, pointer.y];
        let {lineIdx, nearPoint} = aroundWhichLine(pointer);
        if (lineIdx >= 0) {
            line = new fabric.Line(points, {
                stroke: schemeTableau10[lines.length % 10],
                hasControls: false,
                hasBorders: false,
                lockMovementX: false,
                lockMovementY: false,
                hoverCursor: 'default',
                selectable: false,
                strokeDashArray: [5, 5],
            });
            line.set({
                x1: nearPoint.x,
                y1: nearPoint.y
            })
            canvas.add(line);
            isMetricDrawing = true;
            metricStartLine = lineIdx;
            // startJoint = pointIdx;
            return true;
        }else{
            return false;
        }
    }

    function onDblClick(options) {
        //alert('dblclick')
        if (isDown) {
            let pointer = canvas.getPointer(options.e);
            finishLine(pointer)

        }
    };

    function onMouseDown(options) {
        // console.log(isDown);
        // console.log(isMetricDrawing);
        console.log(lines)
        if (isDown) {
            let pointer = canvas.getPointer(options.e);
            finishLine(pointer)

        }else if(isMetricDrawing){
            let pointer = canvas.getPointer(options.e);
            finishDrawingMetric(pointer)

        }
        else {
            let pointer = canvas.getPointer(options.e);
            let success = startLine(pointer)
            if(!success){
                startDrawingMetric(pointer)
            }
        }

    };

    function onMouseOver(e) {
        if (e && e.target && "radius" in e.target && e.target.radius <= 20) {
            e.target.set('fill', styles.hintColor);
            e.target.set("opacity", 0.6)
        }
        canvas.renderAll();
    }

    function onMouseOut(e) {
        if (e && e.target && "radius" in e.target && e.target.radius <= 20) {
            e.target.set('fill', styles.hintColor);
            e.target.set("opacity", 0)
        }
        canvas.renderAll();
    }

    function onMouseMove(o) {
        if (!isDown && !isMetricDrawing) return;
        let pointer = canvas.getPointer(o.e);
        line.set({
            x2: pointer.x,
            y2: pointer.y
        });
        canvas.requestRenderAll();
    }
    //end mouse:move
    canvas.on('mouse:down', onMouseDown);
    canvas.on('mouse:dblclick', onDblClick);
    canvas.on('mouse:move', onMouseMove)
    canvas.on('mouse:over', onMouseOver);
    canvas.on('mouse:out', onMouseOut);
}

export {onDrawLine}

