import React, {useEffect, useRef, useState} from 'react';
import * as d3 from "d3"
import {scaleBand, scaleLinear} from "d3-scale"
import {axisLeft} from "d3";
import {VegaLite, Vega} from "react-vega";
import {describeColumn} from "./utils";

export default function BarDistribution({data, column, height, imageFilter, columnFilters, columnDescription}){
    let table = {
        table: data
    }

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
    let columnDesc = describeColumn(column);
    let spec = {
        height: height,
        width: height,
        "data": {"name": "table"},
        title: "Distribution of " + columnDescription.desc,
        "layer": [
        {
            "params": [{
                "name": "brush",
                "select": {"type": "interval", "encodings": ["x"]}
            }],
            "mark": "bar",
            "encoding": {
                "x": {"field": columnName, "bin": true, scale:{
                    domain
                    }},
                "y": {"aggregate": "count"}
            }
        },
        {
            // "transform": [{"filter": "datum.#1 > 30"}],
            "transform": [{"filter": {"field": "file_id", "oneOf": imageFilter}}],
            "mark": "bar",
            "encoding": {
                "x": {"field": columnName, "bin": true, scale: {
                    domain
                    }},
                "y": {"aggregate": "count"},
                "color": {"value": "goldenrod"}
            }
        },

    ],

    }

    return (
        <div>
            <Vega spec={spec} data={table} renderer={"svg"}/>
        </div>
    )

}