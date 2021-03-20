import React, {useEffect, useRef, useState} from 'react';
import * as d3 from "d3"
import {scaleBand, scaleLinear} from "d3-scale"
import {axisLeft} from "d3";
import {VegaLite} from "react-vega";

export default function BarDistribution({data, columnName, height}){
    let table = {
        table: data
    }
    let spec = {
        height: height,
        width: height,
        "data": {"name": "table"},
        title: "Distribution of " + columnName,
        "layer": [
        {
            "params": [{
                "name": "brush",
                "select": {"type": "interval", "encodings": ["x"]}
            }],
            "mark": "bar",
            "encoding": {
                "x": {"field": columnName, "bin": true},
                "y": {"aggregate": "count"}
            }
        },
        // {
        //     "transform": [{"filter": {"param": "brush"}}],
        //     "mark": "bar",
        //     "encoding": {
        //         "x": {"field": columnName, "bin": true},
        //         "y": {"aggregate": "count"},
        //         "color": {"value": "goldenrod"}
        //     }
        // }
    ]
    }

    return (
        <div>
            <VegaLite spec={spec} data={table} />
        </div>
    )

}