import React, {useEffect, useRef, useState} from 'react';
import * as d3 from "d3"
import {scaleBand, scaleLinear} from "d3-scale"
import {axisLeft} from "d3";
import {VegaLite, Vega} from "react-vega";
import {describeColumn} from "./utils";
import Typography  from "@material-ui/core/Typography";
import Card from '@material-ui/core/Card';
import Slider from '@material-ui/core/Slider';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
function valuetext(value) {
    return `${value}`;
}

export default function BarDistribution({data, column, imageFilter, columnFilters, columnDescription, width, height}){
    let table = {
        table: data
    }
    const [binSize, setBinSize] = useState(5);
    const [autoScale, setAutoScale] = useState(false);
    useEffect(() => {
        let resizeTimer;
        const handleResize = () => {
            // clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function () {
                // setWidth(window.innerWidth);
                // setHeight(window.innerHeight);
                // canvasObj?.setWidth(window.innerWidth)
            }, 300);
            // setHeight(window.innerHeight);
            console.log(window.innerHeight)
            console.log("hello wsorld");
        };
        window.addEventListener("resize", handleResize);
        return () => {
            window.removeEventListener("resize", handleResize);
        };
    }, []);

    let padding = {
        top: 25,
        bottom: 8,
        right: 15,
        left: 5
    }
    let containerStyle = {
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",

    }
    let headerContainerStyle = {
        background: "#F1F1F1",
        width: "100%"
    }
    let visContainerStyle = {
        // paddingTop: padding.top,
        paddingTop: padding.top + "px",
        paddingBottom: padding.bottom + "px",
        paddingRight: padding.right + "px",
        paddingLeft: padding.left + "px",
        display: "flex",
        flexDirection: "column"
    }
    let controlPanelStyle = {
        display: "flex",
        flexDirection: "row",
        marginLeft: "50px",
        justifyContent: "space-evenly",
        marginBottom: "5px"
    }
    let controlSwitchStyle = {

    }
    let controlSliderStyle = {
        width: "90px"
    };
    let switchStyle = {
        display: "flex",
        flexDirection: "column"
    }
    let sliderContainerStyle = {
        display: "flex",
        flexDirection: "column"
    }
    let headerStyle = {
        paddingLeft: "15px",
        marginTop: "10px",
        marginBottom: "5px",
        textAlign: "left"
        // fontSize: 14,
    };
    let domain = [0, 180]
    let columnName = column.key;
    let thisFilter = columnFilters.filter(d=>d.dimension.key === columnName);
    let extent = []
    if(column.type === "Angle"){
        domain = [0, 180]
    }else{
        domain = [0, 100]
    }
    if(thisFilter.length === 1){
        extent = thisFilter[0].extent;
    }
    // mark-rect role-mark
    useEffect(()=>{
        let svg = d3.select(".chart-wrapper").select("svg");
        let axisSelection = svg.selectAll(".role-axis").nodes()
        let chartContent = axisSelection[0];
        let xAxisG = axisSelection[1];
        let yAxisG = axisSelection[2]
        let chartWidth = chartContent.getBoundingClientRect().width;
        let translateX = yAxisG.getBoundingClientRect().width;
        let translateY = yAxisG.getBoundingClientRect().height + yAxisG.getBoundingClientRect().y - svg.node().getBoundingClientRect().y;
        // let marginLeft = chartContent[1].node().getBoundingClientRect().width;
        let xScale = scaleLinear().domain([0, 180]).range([0, chartWidth]);
        svg.selectAll(".brush-container").remove();
        let brushContainer = svg.append("g").attr("class","brush-container").attr("transform", "translate(" + translateX + "," + translateY + ")");
        brushContainer.empty();
        brushContainer.append("g")
            .append("circle")
            .attr("class", "brush-anchor")
            .attr("cx", xScale(extent[0]))
            .attr("cy", 0)
            .attr("r", 5)
            .attr("fill", "None")
            .attr("stroke", "goldenrod")
        brushContainer.append("g")
            .append("circle")
            .attr("class", "brush-anchor")
            .attr("cx", xScale(extent[1]))
            .attr("cy", 0)
            .attr("r", 5)
            .attr("fill", "None")
            .attr("stroke", "goldenrod")

        // let brushAnchors = brushContainer.selectAll("g")
        //     .data(extent)
        //     .enter().append("g")
        //     .append("circle")
        //



    },[columnFilters, columnName])

    let getXEncoding = (useAuto) => {
        if(!useAuto){
            return {
                "field": columnName, "bin": {step: binSize}, scale: {
                    domain
                }
            }
        }else{
            return {
                "field": columnName, "bin": {step: binSize}
            }
        }
    }
    let xEncoding = getXEncoding(autoScale);

    let columnDesc = describeColumn(column);
    let spec = {
        height: height - padding.top - padding.bottom,
        width: width - padding.left - padding.right,
        "data": {"name": "table"},
        // title: "Distribution of " + columnDescription.desc,
        "layer": [
        {
            "params": [
                // {
                //     "name": "brush",
                //     "select": {"type": "interval", "encodings": ["x"]}
                // },
                // {
                //     "name": "binsize",
                //     value: 15,
                //     // "select": {"type": "point", "fields": ["Origin"]},
                //     "bind": {"input": "range", "min": 2, "max": 30}
                // }
            ],
            "mark": "bar",
            "encoding": {
                "x": xEncoding,
                "y": {"aggregate": "count"}
            }
        },
        {
            // "transform": [{"filter": "datum.#1 > 30"}],
            "transform": [{"filter": {"field": "file_id", "oneOf": imageFilter}}],
            "mark": "bar",
            "encoding": {
                "x": xEncoding,
                "y": {"aggregate": "count"},
                "color": {"value": "goldenrod"}
            }
        },

    ],

    }



    let handleScaleChange = ()=>{
        setAutoScale(!autoScale);
    }
    let handleSliderChange = (e, newvalue) =>{
        setBinSize(newvalue)
    }
    return (
        <Card style={containerStyle}>
            <div style={headerContainerStyle}>
                <Typography style={headerStyle} variant="h5" component="h4">
                    { columnDescription.desc}
                </Typography>
            </div>

            <div style={visContainerStyle}>

                <Vega spec={spec} data={table} renderer={"svg"}/>

            </div>
            <div style={controlPanelStyle}>
                {/*<div style={switchStyle}>*/}
                    <FormControlLabel style={controlSwitchStyle}
                                      control={
                                          <Switch
                                              size={"small"}
                                              checked={autoScale}
                                              onChange={() => handleScaleChange()}
                                              name="auto scale"
                                              color="primary"
                                          />
                                      }
                                      // label={""}
                                      label="Auto Scale"
                    />
                    {/*<span> Adaptive Scale </span>*/}
                {/*</div>*/}


                <div style={sliderContainerStyle}>
                    <Slider
                        style={controlSliderStyle}
                        value={binSize}
                        getAriaValueText={valuetext}
                        aria-labelledby="discrete-slider"
                        valueLabelDisplay="auto"
                        onChange={handleSliderChange}
                        step={1}
                        marks
                        min={1}
                        max={30}
                    />
                    <span> Bin Size</span>
                </div>


            </div>
        </Card>

    )

}