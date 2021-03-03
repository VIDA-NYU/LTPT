import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
import React, {useEffect, useState} from 'react';
import {fabric} from "fabric"

import './App.css';
import {onDrawLine} from './interaction';
import {IEvent} from "fabric/fabric-impl";
import {type} from "os";

interface BodyJoint {
    name: string,
    coords: Array<number>
}


function CanvasComponent({args}: ComponentProps) {
    const [canvas, setCanvas] = useState(new fabric.Canvas(""))
    useEffect(() => {
        const c = new fabric.Canvas("canvas", {
            enableRetinaScaling: false,
        })
        const canvas = new fabric.Canvas("interaction-canvas", {
            enableRetinaScaling: false,
        })
        setCanvas(canvas);
        Streamlit.setFrameHeight()
        let circleFill = "#dbe2ef"
        let circleStroke = "#666"

        function makeCircle(coords: Array<number>){
            let circle = new fabric.Circle({
                radius: 5, fill: circleFill, left: coords[0], top: coords[1], stroke: circleStroke, selectable:false,
                originX: "center",
                originY: "center",
            });
            return circle
        }
        function makeLine(coords: Array<number>) {
            // coords = coords.map(d=>d+2.5)
            return new fabric.Line(coords, {
                fill: 'red',
                stroke: 'red',
                strokeWidth: 5,
                selectable: false,
                evented: false,
                originX: "center",
                originY: "center",
            });
        }
        function makeCoveringCircle(coords: Array<number>){
            let circle = new fabric.Circle({
                radius: 20, fill: circleFill, left: coords[0], top: coords[1], stroke: circleStroke, selectable:false,
                originX: "center",
                originY: "center",
                opacity: 0.01,
            });
            return circle
        }

        const bodyJoints: BodyJoint[] = [
            {
                name: "head",
                coords: [200, 50]
            },
            {
                name: "leftHand",
                coords: [100, 100]
            },
            {
                name: "rightHand",
                coords: [300, 100]
            },
            {
                name: "leftFoot",
                coords: [100, 300]
            },
            {
                name: "rightFoot",
                coords: [300, 300]
            },
            {
                name: "upSpine",
                coords: [200, 150]
            },
            {
                name: "downSpine",
                coords: [200, 250]
            }
        ]

        let pointCoords = {
            "leftHand": [100, 100],
            "rightHand": [300, 100],
            "leftFoot": [100, 300],
            "rightFoot": [300, 300],
            "head": [200, 50],
            "upSpine": [200, 150],
            "downSpine": [200, 250]
        }


        // let leftHand = makeCircle([100, 100])
        // let rightHand = makeCircle([300, 100])

        // let leftFoot = makeCircle([100, 300])

        // let rightFoot = makeCircle([300, 300])
        // let upSpine = makeCircle([200, 150])
        // let downSpine = makeCircle([200, 250]);
        let line = makeLine([200, 50, 200, 150]),
            line2 = makeLine([100, 100, 200, 150]),
            line3 = makeLine([300, 100, 200, 150]),
            line4 = makeLine([200, 150, 200, 250]),
            line5 = makeLine([100, 300, 200, 250]),
            line6 = makeLine([300, 300, 200, 250]);
        canvas.add(line, line2, line3, line4, line5, line6);
        for (let bodyJoint of bodyJoints){
            let coords = bodyJoint.coords;
            let circle = makeCircle(coords);
            let coverCircle = makeCoveringCircle(coords)
            canvas.add(circle);
            canvas.add(coverCircle);

        }
        // canvas.add(headCircle);
        // canvas.add(leftHand)
        // canvas.add(rightHand)
        // canvas.add(leftFoot)
        // canvas.add(rightFoot)
        // canvas.add(upSpine)
        // canvas.add(downSpine)

        onDrawLine(canvas, bodyJoints);

    }, [])

    let canvasHeight = 400;
    let canvasWidth = 400;
    return (
        <div>
            <canvas
                id="interaction-canvas"
                width={canvasWidth}
                height={canvasHeight}
            />

        </div>
    );
}

export default withStreamlitConnection(CanvasComponent);
