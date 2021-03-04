import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
import React, {useEffect, useState} from 'react';
import {fabric} from "fabric"

import './App.css';
import {onDrawLine} from './interaction';
import {IEvent} from "fabric/fabric-impl";
import {type} from "os";
import {keypointConfigs, auxiliaryKeypointConfigs, auxiliaryLineConfigs, PointConfig, CanvasPoint} from './constants';



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
        let transformCoords = (coords: Array<number>)=>{
            return [coords[0] * 400, coords[1] * 400];
        }
        let keypointsOnCanvas = keypointConfigs.map(d=>{
            return {
                ...d,
                coords: transformCoords(d.coords)
            }
        });

        let allPointConfigs: Array<PointConfig> = [...keypointConfigs, ...auxiliaryKeypointConfigs];
        for (let lineConfig of auxiliaryLineConfigs){
            let src = allPointConfigs.filter(d=>d.name===lineConfig.src)[0];
            let srcCoords = transformCoords(src.coords)
            let dest = allPointConfigs.filter(d=>d.name===lineConfig.dest)[0];
            let destCoords = transformCoords(dest.coords)
            let lineCoords = [...srcCoords, ...destCoords];
            let line = makeLine(lineCoords);
            canvas.add(line);
        }

        // let line = makeLine([200, 50, 200, 150]),
        //     line2 = makeLine([100, 100, 200, 150]),
        //     line3 = makeLine([300, 100, 200, 150]),
        //     line4 = makeLine([200, 150, 200, 250]),
        //     line5 = makeLine([100, 300, 200, 250]),
        //     line6 = makeLine([300, 300, 200, 250]);
        // canvas.add(line, line2, line3, line4, line5, line6);

        for (let keypointConfig of keypointConfigs){
            let coords = keypointConfig.coords;
            coords = [coords[0] * 400, coords[1]*400];
            let circle = makeCircle(coords);
            let coverCircle = makeCoveringCircle(coords)
            canvas.add(circle);
            canvas.add(coverCircle);
        }

        onDrawLine(canvas, keypointsOnCanvas);

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
