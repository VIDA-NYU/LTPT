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
    console.log(dist);
    if (dist > 10) {
        return {isAround: false, dist, lineNearPoint: null}
    } else {
        console.log("sh")
        console.log(nearPoint)
        return {isAround: true, dist, lineNearPoint: nearPoint}
    }
}


function onDrawLine(canvas, fixedPoints) {
    let waterPipePoints = [];
    let waterPipeLines = [];
    let line = null;
    let isDown = false;
    let isMetricDrawing = false;
    let lines = [];
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
        console.log(lines)
        let nearPoint = null;
        for (let i in lines) {
            let lineData = lines[i];
            let src = parseInt(lineData.src);
            let dest = parseInt(lineData.dest);
            let coords = [
                fixedPoints[src].coords[0], fixedPoints[src].coords[1],
                fixedPoints[dest].coords[0], fixedPoints[dest].coords[1]
            ]
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
        console.log(nearPoint);
        console.log("idx")
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
                src: startJoint,
                dest: pointIdx,
                color: schemeTableau10[lines.length]
            }
            lines.push(lineData);
            line = null;
            Streamlit.setComponentValue({
                data: lines
            })
        }
    }

    function startLine(pointer) {
        let points = [pointer.x, pointer.y, pointer.x, pointer.y];
        let pointIdx = aroundWhichPoint(pointer);
        if (pointIdx >= 0) {
            line = new fabric.Line(points, {
                stroke: schemeTableau10[lines.length],
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
            // let lineData = {
            //     src: startJoint,
            //     dest: pointIdx,
            //     color: schemeTableau10[lines.length]
            // }
            // lines.push(lineData);
            line = null;
            let metric = {

            }
            if(lineIdx === metricStartLine){
                metric['type'] = "distance";
                metric['lines'] = [lineIdx]
            }else{
                metric['type'] = "angle";
                metric['lines'] = [lineIdx, metricStartLine]
            }

            Streamlit.setComponentValue({
                data: lines,
                metric
            })
        }
    }
    function startDrawingMetric(pointer){
        let points = [pointer.x, pointer.y, pointer.x, pointer.y];
        let {lineIdx, nearPoint} = aroundWhichLine(pointer);
        if (lineIdx >= 0) {
            line = new fabric.Line(points, {
                stroke: schemeTableau10[lines.length],
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
        if (e && e.target && "radius" in e.target) {
            e.target.set('fill', 'red');
            e.target.set("opacity", 1)
        }
        canvas.renderAll();
    }

    function onMouseOut(e) {
        if (e && e.target && "radius" in e.target) {
            e.target.set('fill', 'red');
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
    }; //end mouse:move
    canvas.on('mouse:down', onMouseDown);
    canvas.on('mouse:dblclick', onDblClick);
    canvas.on('mouse:move', onMouseMove)
    canvas.on('mouse:over', onMouseOver);
    canvas.on('mouse:out', onMouseOut);
    // canvas.on("mouse:down", function (event) {
    //     var pointer = canvas.getPointer(event.e);
    //     var positionX = pointer.x;
    //     var positionY = pointer.y;
    //
    //     let selectedIdx = -1;
    //     let minDist = -1;
    //     // console.log(pointer)
    //     for (let i in fixedPoints) {
    //         let point = fixedPoints[i];
    //         let {clicked, dist} = clickOnPoint(pointer, point);
    //         if (clicked) {
    //             if (minDist < 0 | minDist > dist) {
    //                 dist = minDist;
    //                 selectedIdx = i;
    //             }
    //         }
    //     }
    //     if (selectedIdx < 0) {
    //         return;
    //     }
    //
    //     // Add small circle as an indicative point
    //     var circlePoint = new fabric.Circle({
    //         radius: 5,
    //         fill: "blue",
    //         left: fixedPoints[selectedIdx][0],
    //         top: fixedPoints[selectedIdx][1],
    //         selectable: false,
    //         originX: "center",
    //         originY: "center",
    //         hoverCursor: "auto"
    //     });
    //
    //     canvas.add(circlePoint);
    //
    //     // Store the points to draw the lines
    //     waterPipePoints.push(circlePoint);
    //     console.log(waterPipePoints);
    //     if (waterPipePoints.length > 1) {
    //         // Just draw a line using the last two points, so we don't need to clear
    //         // and re-render all the lines
    //         var startPoint = waterPipePoints[waterPipePoints.length - 2];
    //         var endPoint = waterPipePoints[waterPipePoints.length - 1];
    //
    //         var waterLine = new fabric.Line(
    //             [
    //                 startPoint.get("left"),
    //                 startPoint.get("top"),
    //                 endPoint.get("left"),
    //                 endPoint.get("top")
    //             ],
    //             {
    //                 stroke: "blue",
    //                 strokeWidth: 4,
    //                 hasControls: false,
    //                 hasBorders: false,
    //                 selectable: false,
    //                 lockMovementX: true,
    //                 lockMovementY: true,
    //                 hoverCursor: "default",
    //                 originX: "center",
    //                 originY: "center"
    //             }
    //         );
    //
    //         waterPipeLines.push(waterLine);
    //
    //         canvas.add(waterLine);
    //     }
    // });

}

export {onDrawLine}

