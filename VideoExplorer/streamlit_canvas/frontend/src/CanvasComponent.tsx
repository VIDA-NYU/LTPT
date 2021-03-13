import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
import React, {useEffect, useState} from 'react';
import {fabric} from "fabric"

import './App.css';
import {onDrawLine} from './interaction';
import {IEvent} from "fabric/fabric-impl";
import {type} from "os";
import {keypointConfigs, auxiliaryKeypointConfigs, auxiliaryLineConfigs, PointConfig, CanvasPoint, decorationShapeConfigs, ShapeConfig} from './constants';



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
        let lineColor = "red"
        let skeletonColor = "#393e46"
        let hintColor = "#eeeeee"
        let styles = {
            hintColor, skeletonColor
        }
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
                stroke: skeletonColor,
                strokeWidth: 3,
                selectable: false,
                evented: false,
                originX: "center",
                originY: "center",
            });
        }
        function makeCoveringCircle(coords: Array<number>){
            let circle = new fabric.Circle({
                radius: 20, fill: hintColor, left: coords[0], top: coords[1], stroke: styles.skeletonColor, selectable:false,
                originX: "center",
                originY: "center",
                opacity: 0.01,
            });
            return circle
        }
        function makeMouth(config: ShapeConfig){
            let circle = new fabric.Circle({
                radius: 10,
                fill: undefined,
                left: config.coords[0],
                top: config.coords[1],
                stroke: skeletonColor,
                selectable: false,
                originX: "center",
                originY: "center",
                startAngle: Math.PI / 4,
                endAngle: 3 * Math.PI / 4,
            })
            canvas.add(circle);
        }
        function makeHeadCircle(config: ShapeConfig){
            let circle = new fabric.Circle({
                radius: 36,
                fill: undefined,
                left: config.coords[0],
                top: config.coords[1],
                stroke: skeletonColor,
                selectable: false,
                originX: "center",
                originY: "center"
            })
            canvas.add(circle);
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
        let auxiliaryPointsOnCanvas = auxiliaryKeypointConfigs.map(d=>{
            return {
                ...d,
                coords: transformCoords(d.coords)
            }
        });
        let decorationShapeOnCanvas = decorationShapeConfigs.map(d=>{
            return {
                ...d,
                coords: transformCoords(d.coords),
            }
        })

        for (let config of decorationShapeOnCanvas){
            if (config.type === "mouth"){
                makeMouth(config);
            }else if(config.type === "head"){
                makeHeadCircle(config);
            }
        }
        for (let keypointConfig of keypointConfigs){
            let coords = keypointConfig.coords;
            coords = [coords[0] * 400, coords[1]*400];
            let circle = makeCircle(coords);
            let coverCircle = makeCoveringCircle(coords)
            canvas.add(circle);
            canvas.add(coverCircle);
        }

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




        onDrawLine(canvas, keypointsOnCanvas, styles);

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
