import {fabric} from "fabric";
import {schemeTableau10, schemeSet3} from "d3-scale-chromatic"
import {Streamlit} from 'streamlit-component-lib'
import {keypointConfigs} from "./constants";

const colorScheme = schemeSet3
// const lineColor = "#eb5e0b";
// const lineColor = "#91091e";
// const lineColor = "#a3d2ca"
const lineColor = "#df7861"
const strokeWidth = 2;

function isAroundPoint(pointer, coords) {

    let dist = (pointer.x - coords[0]) * (pointer.x - coords[0]) + (pointer.y - coords[1]) * (pointer.y - coords[1]);
    dist = Math.sqrt(dist)
    if (dist < 10) {
        return {isAround: true, dist}
    } else {
        return {isAround: false, dist}
    }
}

function isAroundLine(pointer, coords) {
    let v1 = [pointer.x - coords[0], pointer.y - coords[1]];
    let v2 = [coords[2] - coords[0], coords[3] - coords[1]]
    let d = Math.sqrt(Math.pow(v2[0], 2) + Math.pow(v2[1], 2));
    let t = (v1[0] * v2[0] + v1[1] * v2[1]) / Math.sqrt(Math.pow(v2[0], 2) + Math.pow(v2[1], 2));
    let lPower = Math.pow(v1[0], 2) + Math.pow(v1[1], 2);
    let dist = Math.sqrt(lPower - Math.pow(t, 2));
    let nearPoint = {
        x: coords[0] + v2[0] * t / d,
        y: coords[1] + v2[1] * t / d
    }

    let vn1 = [nearPoint.x - coords[0], nearPoint.y - coords[1]];
    let vn2 = [nearPoint.x - coords[2], nearPoint.y - coords[3]];
    if (dotProduct(vn1, vn2) > 0) {
        return {isAround: false, dist, lineNearPoint: null};
    }

    if (dist > 10) {
        return {isAround: false, dist, lineNearPoint: null}
    } else {
        return {isAround: true, dist, lineNearPoint: nearPoint}
    }
}

function vectorLen(v) {
    return Math.sqrt(Math.pow(v[0], 2) + Math.pow(v[1], 2));
}

function coordsAdd(a, b) {
    let r = a.map((v, i) => v + b[i]);
    return r;
}

function coordsSubtract(a, b) {
    return a.map((v, i) => v - b[i]);
}

function vectorDivide(a, b) {
    if (b === 0) {
        return [0, 0]
    }
    return a.map(v => v / b);
}

function vectorMultiply(a, b) {
    return a.map(v => v * b);
}

function getAngle(v) {
    let angle = Math.atan2(v[1], v[0]);
    return angle
}

function dotProduct(v1, v2) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}

function calculateArcParamsAttached(origin, dest, lineCoords) {
    let lineVec = coordsSubtract(lineCoords.dest, lineCoords.src);
    let lAddition = 25;
    let edgeL = 50;
    let v = coordsSubtract(dest, origin)

    let dot = dotProduct(lineVec, v);
    let direction = dot > 0 ? 1 : -1;
    let center = lineCoords.dest
    if (dot < 0) {
        center = lineCoords.src;
    }
    let vo = [-v[1], v[0]];
    vo = vectorMultiply(vo, direction)
    vo = vectorDivide(vo, vectorLen(vo));
    let dist = vectorLen(v);
    // L = Math.round(dist/L) * L;
    // let voL = Math.sqrt(L*L - Math.pow(dist/2, 2));
    let voL = vectorLen(v);
    let L = Math.sqrt(Math.pow(voL, 2) + Math.pow(dist / 2, 2))
    L = voL;
    vo = vectorMultiply(vo, voL);
    let pointC = coordsAdd(origin, vectorDivide(v, 2))
    pointC = coordsAdd(pointC, vo);
    pointC = center;
    let v1 = coordsSubtract(origin, pointC);
    let v2 = coordsSubtract(dest, pointC);
    let startAngle = getAngle(v1);
    let endAngle = getAngle(v2);
    return {center: center, startAngle, endAngle, L};
}

function calculateArcParams(origin, dest, lineCoords) {
    let lineVec = coordsSubtract(lineCoords.dest, lineCoords.src);
    let lAddition = 25;
    let edgeL = 50;
    let v = coordsSubtract(dest, origin)

    let dot = dotProduct(lineVec, v);
    let direction = dot > 0 ? 1 : -1;
    let center = lineCoords.dest
    if (dot < 0) {
        center = lineCoords.src;
    }
    let vo = [-v[1], v[0]];
    let dotB = dotProduct(vo, lineVec)
    if (dotB < 0) {
        vo = vectorMultiply(vo, -1)
    }
    vo = vectorMultiply(vo, direction)
    vo = vectorDivide(vo, vectorLen(vo));
    let dist = vectorLen(v);
    // L = Math.round(dist/L) * L;
    // let voL = Math.sqrt(L*L - Math.pow(dist/2, 2));
    let voL = vectorLen(v);
    let L = Math.sqrt(Math.pow(voL, 2) + Math.pow(dist / 2, 2))
    vo = vectorMultiply(vo, voL);
    let pointC = coordsAdd(origin, vectorDivide(v, 2))
    pointC = coordsAdd(pointC, vo);
    let v1 = coordsSubtract(origin, pointC);
    let v2 = coordsSubtract(dest, pointC);
    let startAngle = (dot * dotB > 0) ? getAngle(v1) : getAngle(v2)
    let endAngle = (dot * dotB > 0) ? getAngle(v2) : getAngle(v1);
    return {center: pointC, startAngle, endAngle, L};
}


var lines = [];

function onDrawLine(canvas, fixedPoints, styles, addAMetric) {
    let line = null;
    let isDown = false;
    let isMetricDrawing = false;
    let startJoint = -1;
    let metricStartLine = -1;
    let arc = null;

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

    function aroundWhichLine(pointer) {
        let selectedIdx = -1;
        let minDist = -1;
        let nearPoint = null;
        for (let i in lines) {
            let lineData = lines[i];
            let src = lineData.src;
            let dest = lineData.dest;
            let srcPoint = fixedPoints.filter(d => d.name === src)[0];
            let destPoint = fixedPoints.filter(d => d.name === dest)[0];
            let coords = [
                srcPoint.coords[0], srcPoint.coords[1],
                destPoint.coords[0], destPoint.coords[1]
            ];
            let point = fixedPoints[i];
            let {isAround, dist, lineNearPoint} = isAroundLine(pointer, coords);
            if (isAround) {
                if (minDist < 0 | minDist > dist) {
                    dist = minDist;
                    selectedIdx = i;
                    nearPoint = lineNearPoint

                }
            }
        }
        if (selectedIdx < 0) {
            return {lineIdx: selectedIdx, nearPoint: null};
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
                color: "#eb5e0b"
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
                stroke: lineColor,
                strokeWidth: strokeWidth,
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
        } else {
            return false;
        }

    }

    function finishDrawingMetric(pointer) {
        let {lineIdx, nearPoint} = aroundWhichLine(pointer);
        // canvas.selectable = false;
        isMetricDrawing = false;
        if (lineIdx < 0) {

            alert("Please select a line");
            arc.set(
                {"stroke": null}
            );
            line = null;

        } else {
            if (lineIdx === metricStartLine) {
                line.set({
                    stroke: colorScheme[lines.length % 10],
                    x1: fixedPoints.filter(d => d.name === lines[metricStartLine].src)[0].coords[0],
                    y1: fixedPoints.filter(d => d.name === lines[metricStartLine].src)[0].coords[1],
                    x2: fixedPoints.filter(d => d.name === lines[metricStartLine].dest)[0].coords[0],
                    y2: fixedPoints.filter(d => d.name === lines[metricStartLine].dest)[0].coords[1],
                })
                arc.set({
                    stroke: null
                })
            } else if (lineIdx >= 0) {
                line.setCoords();
                line.set({
                    x2: nearPoint.x,
                    y2: nearPoint.y
                })
                let srcCoords = fixedPoints.filter(d => d.name === lines[metricStartLine].src)[0].coords
                let destCoords = fixedPoints.filter(d => d.name === lines[metricStartLine].dest)[0].coords
                let lineCoords = {
                    src: srcCoords,
                    dest: destCoords
                }
                let {
                    center,
                    startAngle,
                    endAngle,
                    L
                } = calculateArcParams([line.x1, line.y1], [nearPoint.x, nearPoint.y], lineCoords);
                arc.set({
                    radius: L,
                    left: center[0],
                    top: center[1],

                    startAngle, endAngle
                })
            }

            let metric = {}
            if (lineIdx === metricStartLine) {
                metric['type'] = "Distance";
                metric['lines'] = [lineIdx].map(d => lines[d])
            } else {
                metric['type'] = "Angle";
                metric['lines'] = [lineIdx, metricStartLine].map(d => lines[d])
            }
            addAMetric({
                type: metric['type'],
                name: "default name",
                lines: metric['lines'],
                visibility: true

            })
        }
        isMetricDrawing = false;
        line = null;

    }

    function startDrawingMetric(pointer) {
        let points = [pointer.x, pointer.y, pointer.x, pointer.y];
        let {lineIdx, nearPoint} = aroundWhichLine(pointer);
        if (lineIdx >= 0) {
            line = new fabric.Line(points, {
                stroke: null,
                strokeWidth: 3,
                hasControls: false,
                hasBorders: false,
                lockMovementX: false,
                lockMovementY: false,
                hoverCursor: 'default',
                selectable: false,
                strokeDashArray: [5, 5],
            });
            arc = new fabric.Circle({
                radius: 12,
                fill: null,
                left: nearPoint.x,
                top: nearPoint.y,
                stroke: colorScheme[lines.length % 10],
                strokeWidth: strokeWidth,
                selectable: false,
                originX: "center",
                originY: "center",
                startAngle: 14 * Math.PI / 8,
                endAngle: 14 * Math.PI / 8,
            })
            line.set({
                x1: nearPoint.x,
                y1: nearPoint.y
            })
            canvas.add(line);
            canvas.add(arc);
            isMetricDrawing = true;
            metricStartLine = lineIdx;
            // startJoint = pointIdx;
            return true;
        } else {
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

        } else if (isMetricDrawing) {
            let pointer = canvas.getPointer(options.e);
            finishDrawingMetric(pointer)

        } else {
            let pointer = canvas.getPointer(options.e);
            let success = startLine(pointer)
            if (!success) {
                startDrawingMetric(pointer)
            }
        }

    };

    function onMouseOver(e) {
        if (e && e.target && "radius" in e.target && e.target.radius == 20) {
            e.target.set('fill', styles.hintColor);
            e.target.set("opacity", 0.6)
        }
        canvas.renderAll();
    }

    function onMouseOut(e) {
        if (e && e.target && "radius" in e.target && e.target.radius == 20) {
            e.target.set('fill', styles.hintColor);
            e.target.set("opacity", 0)
        }
        canvas.renderAll();
    }

    function onMouseMove(o) {
        if (!isDown && !isMetricDrawing) return;
        if (isMetricDrawing) {
            let pointer = canvas.getPointer(o.e);
            let pointCoords = [pointer.x, pointer.y];
            let srcCoords = fixedPoints.filter(d => d.name === lines[metricStartLine].src)[0].coords
            let destCoords = fixedPoints.filter(d => d.name === lines[metricStartLine].dest)[0].coords
            let lineCoords = {
                src: srcCoords,
                dest: destCoords
            }
            let {center, startAngle, endAngle, L} = calculateArcParams([line.x1, line.y1], pointCoords, lineCoords)
            line.set({
                x2: pointer.x,
                y2: pointer.y
            });
            arc.set({
                radius: L,
                left: center[0],
                top: center[1],
                startAngle, endAngle
            })
            canvas.requestRenderAll();
            return;
        }
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

