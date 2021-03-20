import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
import React, {useEffect, useRef, useState, useCallback} from 'react';
import {fabric} from "fabric"
import CSS from 'csstype';

import './App.css';
import {onDrawLine} from './interaction';
import {MetricTable} from "./MetricTable"
import {IEvent} from "fabric/fabric-impl";
import {type} from "os";
import {keypointConfigs, auxiliaryKeypointConfigs, auxiliaryLineConfigs, PointConfig, CanvasPoint, decorationShapeConfigs, ShapeConfig} from './constants';

enum MetricType {
    Angle="Angle", Distance="Distance"
}
interface Line {
    src: string,
    dest: string,
    color: string
}
interface Metric {
    name: string,
    type: MetricType,
    lines: Array<Line>,
    visibility: boolean
    id?: string

}

function CanvasComponent({args} : ComponentProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvasObj, setCanvas] = useState(new fabric.Canvas("interaction-canvas", {
        enableRetinaScaling: false,
    }))
    const [draw, setDraw] = useState(0);
    const [metrics, setMetrics] = useState<Array<Metric>>([]);
    const [metricCount, setMetricCount] = useState<number>(0)
    let canvasHeight = 400;
    let canvasWidth = 400;
    let updateStatus = (metric: Metric) => {
        metric.id = "#" + metrics.length.toString();
        setMetrics([
            ...metrics, metric
        ])
        setMetricCount(metricCount + 1);
        Streamlit.setComponentValue({
            // data: lines,
            metrics: [...metrics, metric]
        })
    }
    let updateMetricStatus = (newData: Array<Metric>)=>{
        setMetrics(newData);
        Streamlit.setComponentValue({
            // data: lines,
            metrics: newData
        })
    }
    let skeletonColor = "#393e46"
    let hintColor = "#eeeeee"
    let styles = {
        hintColor, skeletonColor
    }
    let transformCoords = (coords: Array<number>)=>{
        return [coords[0] * canvasWidth, coords[1] * canvasHeight];
    }
    let keypointsOnCanvas = keypointConfigs.map(d=>{
        return {
            ...d,
            coords: transformCoords(d.coords)
        }
    });
    useEffect(() => {
        const c = new fabric.Canvas("canvas", {
            enableRetinaScaling: false,
        })
        const canvas = new fabric.Canvas("interaction-canvas", {
            enableRetinaScaling: false,
        })
        // let canvas = canvasObj
        setCanvas(canvas);
        Streamlit.setFrameHeight()
        let circleFill = "#dbe2ef"
        let circleStroke = "#666"
        let lineColor = "red"

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
        function makeEye(config: ShapeConfig){

            let circle = new fabric.Circle({
                radius: 12,
                fill: undefined,
                left: config.coords[0],
                top: config.coords[1],
                stroke: skeletonColor,
                selectable: false,
                originX: "center",
                originY: "center",
                startAngle: 10 * Math.PI / 8,
                endAngle: 14 * Math.PI / 8,
            })
            canvas.add(circle);
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
            }else if(config.type === "eye"){
                makeEye(config)
            }
        }
        for (let keypointConfig of keypointConfigs){
            let coords = keypointConfig.coords;
            coords = [coords[0] * canvasWidth, coords[1]*canvasHeight];
            let circle = makeCircle(coords);

            canvas.add(circle);
            if(keypointConfig.clickable){
                let coverCircle = makeCoveringCircle(coords);
                canvas.add(coverCircle);
            }
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


        // onDrawLine(canvas, keypointsOnCanvas, styles, updateStatus);

    }, []);
    useEffect(()=>{
        onDrawLine(canvasObj, keypointsOnCanvas.filter(d=>d.clickable), styles, updateStatus)
    }, [canvasObj, metrics]);
    const div = useCallback(node => {
        if (node !== null) {
            console.log(node.getBoundingClientRect().height);
            console.log(node.getBoundingClientRect().width);
        }
    }, []);

    // useEffect(()=>{
    //     canvasRef.clear();
    // }, [args['operation']])


    const containerStyles: CSS.Properties = {
        display: "flex",
        flexDirection: "row",
        width: "800",
    }
    const tableStyles: CSS.Properties = {
        width: "400"
    }
    const canvasContainerStyles: CSS.Properties = {
        width: canvasWidth.toString(),
        height: canvasHeight.toString()
    }
    return (
        <div ref={div} style={containerStyles}>
            <div style={canvasContainerStyles}>
                <canvas
                    id="interaction-canvas"
                    ref={canvasRef}
                    width={canvasWidth}
                    height={canvasHeight}
                />
            </div>

            <div style={tableStyles}>
                <MetricTable metrics={metrics} setData={updateMetricStatus}></MetricTable>
            </div>

        </div>
    );
}

export default withStreamlitConnection(CanvasComponent);
// export default CanvasComponent;