import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
import React, {useEffect, useRef, useState, useCallback, CSSProperties} from 'react';
import {fabric} from "fabric"
import CSS from 'csstype';
import bodyImage from "./body.png"
// import './App.css';
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
    // const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvasObj, setCanvas] = useState<fabric.Canvas>();
    // const [backgroundCanvas, setBackgroundCanvas] = useState<fabric.Canvas>();
    // const [draw, setDraw] = useState(0);
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
    // let skeletonColor = "#393e46"
    let skeletonColor = "030303"
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
    useEffect(()=>{
        // const c = new fabric.Canvas("background-canvas", {
        //     enableRetinaScaling: false
        // });
        // let image = new Image()
        // // image.src = "/logo192.png"
        // // image.src = "/logo512.png";
        // image.src = "/body.png";
        // let scale = 340 / 512;
        // let tmp = new fabric.Image(image, {});
        // tmp.set({
        //     scaleX: scale,
        //     scaleY: scale,
        //     left: canvasWidth / 2 - 170
        // })
        // c.add(tmp);
        //
        //
        // const imageData = c
        //     .getContext()
        //     .createImageData(canvasWidth, canvasHeight)
        // // imageData.data.set(bodyImage)
        // // backgroundCanvas.getContext().putImageData(imageData, 0, 0)
        //
        // setBackgroundCanvas(c);
    }, [])
    useEffect(()=>{
        Streamlit.setFrameHeight()

    })
    useEffect(() => {

        const canvas = new fabric.Canvas("interaction-canvas", {
            enableRetinaScaling: false,
        })
        // let canvas = canvasObj
        setCanvas(canvas);
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


        for (let keypointConfig of keypointConfigs){
            let coords = keypointConfig.coords;
            coords = [coords[0] * canvasWidth, coords[1]*canvasHeight];
            let circle = makeCircle(coords);

            if(keypointConfig.clickable){

                canvas.add(circle);
                let coverCircle = makeCoveringCircle(coords);
                canvas.add(coverCircle);

            }
        }
        // canvas.clear()

        let allPointConfigs: Array<PointConfig> = [...keypointConfigs, ...auxiliaryKeypointConfigs];
        // for (let lineConfig of auxiliaryLineConfigs){
        //     let src = allPointConfigs.filter(d=>d.name===lineConfig.src)[0];
        //     let srcCoords = transformCoords(src.coords)
        //     let dest = allPointConfigs.filter(d=>d.name===lineConfig.dest)[0];
        //     let destCoords = transformCoords(dest.coords)
        //     let lineCoords = [...srcCoords, ...destCoords];
        //     let line = makeLine(lineCoords);
        //     canvas.add(line);
        // }


        // onDrawLine(canvas, keypointsOnCanvas, styles, updateStatus);

    }, []);
    useEffect(()=>{
        if (!canvasObj){
            return;
        }
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
        width: "550px",
        // overflowY: "scroll"
    }
    const canvasContainerStyles: CSS.Properties = {
        width: canvasWidth.toString() + "px",
        height: canvasHeight.toString() + "px",
        marginTop: "100",
        paddingTop: "100px"
    }
    let canvasMargin = {
        top: 20
    }
    let imgStyles: CSSProperties = {
        position: "absolute",
        top: 0,
        left: canvasWidth / 2 - 170,
        transform: "translate(" + (canvasWidth/2 ) + ", 0)",
        zIndex: 1,
    }
    return (
        <div ref={div} style={containerStyles}>
            <div style={canvasContainerStyles}>

                <div
                    style={{
                        position: "absolute",
                        top: canvasMargin.top,
                        left: 0,
                        zIndex: 1,
                        width: canvasWidth,
                        height: canvasWidth
                    }}
                >

                    <img src={"/body.png"} width={340} style={imgStyles} />
                {/*<canvas*/}
                {/*    // id={"interaction-canvas"}*/}
                {/*    id={"background-canvas"}*/}
                {/*    width={canvasWidth}*/}
                {/*    height={canvasHeight}*/}
                {/*>*/}

                {/*</canvas>*/}
                </div>
                <div
                    style={{
                        position: "absolute",
                        top: canvasMargin.top,
                        left: 0,
                        zIndex: 12,
                    }}
                >
                    <canvas
                    id={"interaction-canvas"}
                        // id={"background-canvas"}
                    // ref={canvasRef}
                    width={canvasWidth}
                    height={canvasHeight}
                />
                </div>
            </div>

            <div style={tableStyles}>
                <MetricTable metrics={metrics} setData={updateMetricStatus}></MetricTable>
            </div>

        </div>
    );
}

export default withStreamlitConnection(CanvasComponent);
// export default CanvasComponent;