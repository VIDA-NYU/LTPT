import React, {useEffect, useRef, useState} from 'react';
import * as d3 from "d3"
import {scaleBand, scaleLinear} from "d3-scale"
import {axisLeft} from "d3";
import "./ParallelCoordinates.css"
import {renderQueue} from "./utils";
import Typography from "@material-ui/core/Typography";
import Card from "@material-ui/core/Card";
// import {} from "d3"
function hashColumns(cols){
    return cols.map(c=>c.key).join();
}
// const chartColor = "#a3d2ca"
const chartColor = "#df7861"
export default function ParallelCoordinates({data, dimensions, setColumnFocus, onSetImageFilter, highlightImage, height, width, setColumnFilters, columnDescriptions}) {
    const [canvas, setCanvas] = useState(null);
    const [ctx, setCtx] = useState(null);
    // let height = 400;
    let margin = {
        left: 50,
        right: 10,
        top: 50,
        bottom: 5
    }

    let innerHeight = height - margin.top - margin.bottom;
    let innerWidth = width - margin.left - margin.right
    let xScale = d3.scaleBand()
        .domain(dimensions.map(d => d.key))
        .range([0, innerWidth * (dimensions.length + 1) / dimensions.length ]);
    let yAxis = d3.axisLeft();

    const devicePixelRatio = 1;
    console.log(data)
    console.log(dimensions)
    let dimensionValues = dimensions.map(d=>{
        return {
            key: d.key,
            values: data.map(item=>item[d.key])
        }
    })
    console.log(dimensionValues);

    // data.reduce((acc, c)=>{
    //     if()
    // }, {})
    let types = {
        "Number": {
            key: "Number",
            coerce: function (d) {
                return +d;
            },
            extent: d3.extent,
            domain: d=> {
                let count = dimensionValues.filter(count=>count.key===d.key)[0];
                return [0, d3.max(count.values)*1.1]
            },
            within: function (d, extent, dim) {
                return extent[0] <= dim.scale(d) && dim.scale(d) <= extent[1];
            },
            defaultScale: d3.scaleLinear().range([innerHeight, 0])
        },
        "Angle": {
            key: "Angle",
            coerce: function (d) {
                return +d;
            },
            extent: (data)=>[0, 180],
            domain: d => [0, 180],
            within: function (d, extent, dim) {
                return extent[0] <= dim.scale(d) && dim.scale(d) <= extent[1];
            },
            defaultScale: d3.scaleLinear().range([innerHeight, 0])
        },
        "String": {
            key: "String",
            coerce: String,
            extent: function (data) {
                return data.sort();
            },
            within: function (d, extent, dim) {
                return extent[0] <= dim.scale(d) && dim.scale(d) <= extent[1];
            },
            defaultScale: d3.scalePoint().range([0, innerHeight])
        },
        "Date": {
            key: "Date",
            coerce: function (d) {
                return new Date(d);
            },
            extent: d3.extent,
            within: function (d, extent, dim) {
                return extent[0] <= dim.scale(d) && dim.scale(d) <= extent[1];
            },
            defaultScale: d3.scaleTime().range([innerHeight, 0])
        }
    };

    // let container = d3.select("body").append("div")
    //     .attr("class", "parcoords")
    //     .style("width", width + margin.left + margin.right + "px")
    //     .style("height", height + margin.top + margin.bottom + "px");
    dimensions = dimensions.map(d => {
        console.log(d);
        let yScale = scaleLinear().domain(types[d.type].domain(d)).range([0, innerHeight])
        return {
            ...d,
            scale: yScale,
            type: types[d.type]
        }
    });
    let columnHash = hashColumns(dimensions);
    function project(d) {
        return dimensions.map(function (p, i) {
            // check if data element has property and contains a value

            if (
                !(p.key in d) ||
                d[p.key] === null
            ) return null;

            return [xScale(p.key), p.scale(d[p.key])];
        });
    };
    function draw(d) {
        // ctx.strokeStyle = "#df7861";
        ctx.beginPath();
        var coords = project(d);
        coords.forEach(function (p, i) {
            // this tricky bit avoids rendering null values as 0
            if (p === null) {
                // this bit renders horizontal lines on the previous/next
                // dimensions, so that sandwiched null values are visible
                if (i > 0) {
                    var prev = coords[i - 1];
                    if (prev !== null) {
                        ctx.moveTo(prev[0], prev[1]);
                        ctx.lineTo(prev[0] + 6, prev[1]);
                    }
                }
                if (i < coords.length - 1) {
                    var next = coords[i + 1];
                    if (next !== null) {
                        ctx.moveTo(next[0] - 6, next[1]);
                    }
                }
                return;
            }

            if (i == 0) {
                ctx.moveTo(p[0], p[1]);
                return;
            }

            ctx.lineTo(p[0], p[1]);
        });
        ctx.stroke();
    }

    const [render, setRender] = useState(null);
    // let color =
    // let yScales = dimensions

    // let canvas = d3.select("#parcoords-canvas");
    useEffect(()=> {
        let _render = renderQueue(draw).rate(30);
        setRender(_render);
        let _canvas = d3.select("#parcoords-canvas")
        setCanvas(_canvas);
        let ctx = _canvas.node().getContext("2d");
        setCtx(ctx);
    }, [columnHash])
    useEffect(() => {
        let render = renderQueue(draw).rate(30);
        if (ctx === null){
            return
        }
        if (render === undefined){
            return ;
        }
        let svg = d3.select("#parcoords-svg")
            .attr("width", innerWidth + margin.left + margin.right)
            .attr("height", innerHeight + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");



        ctx.globalCompositeOperation = 'darken';
        ctx.globalAlpha = 0.15;
        ctx.lineWidth = 1.5;
        ctx.scale(devicePixelRatio, devicePixelRatio);

        let axes = svg.selectAll(".axis")
            .data(dimensions)
            .enter().append("g")
            .attr("class", function (d) {
                return "axis" // + d.key.replace(/ /g, "_");
            })
            .attr("transform", function (d, i) {
                return "translate(" + xScale(d.key) + ")";
            });


        axes.append("g").each(function (d) {
            d3.select(this).call(axisLeft(d.scale))
        }).append("text")
            .attr("class", "title")
            .attr("text-anchor", "start")
            .text(function (d, i) {
                return columnDescriptions[i].desc;
            });
        axes.on('mouseover', function (e, d) {
                setColumnFocus(d.key);
            d3.select(this).transition()
                .duration('50')
                .attr('opacity', '.85')
            })
            .on('mouseout', function (d, i) {

            });

        axes.append("g")
            .attr("class", "brush")
            .each(function (d) {
                d3.select(this).call(d.brush = d3.brushY()
                    .extent([[-10, 0], [10, height]])
                    .on("start", brushstart)
                    .on("brush", brush)
                    .on("end", brush)
                )
            })
            .selectAll("rect")
            .attr("x", -8)
            .attr("width", 16);
        d3.selectAll(".axis.pl_discmethod .tick text")
            .style("fill", "steelblue");

        function brushstart(event) {
            // d3.event.sourceEvent.stopPropagation();
        }

        function brush() {
            render.invalidate();

            var actives = [];
            svg.selectAll(".axis .brush")
                .filter(function (d) {
                    return d3.brushSelection(this);
                })
                .each(function (d) {
                    actives.push({
                        dimension: d,
                        extent: d3.brushSelection(this)
                    });
                });

            let selected = data.filter(function (d) {
                if (actives.every(function (active) {
                    let dim = active.dimension;
                    // test if point is within extents for each active brush
                    return dim.type.within(d[dim.key], active.extent, dim);
                })) {
                    return true;
                }
            });

            onSetImageFilter(selected.map(d=>d.file_id),actives)
            // ctx.strokeStyle = "#df7861";
            ctx.strokeStyle = chartColor;
            ctx.clearRect(0, 0, width, height);
            ctx.globalAlpha = d3.min([0.85 / Math.pow(selected.length, 0.3), 1]);
            render(selected);
            // ctx.globalAlpha = d3.min([0.85 / Math.pow(selected.length, 0.3), 0.05]);
            // ctx.strokeStyle = "black";
            // render(data)
            // ctx.strokeStyle = "#df7861";
        }



        // ctx.strokeStyle = "#df7861";
        ctx.strokeStyle = chartColor;
        ctx.clearRect(0, 0, width, height);
        ctx.globalAlpha = d3.min([1.15 / Math.pow(data.length, 0.3), 1]);
        render(data);

    }, [ctx, render, canvas]);

    useEffect(()=>{
        if (ctx === null || highlightImage===null){
            return
        }
        let render = renderQueue(draw).rate(30);
        ctx.strokeStyle = "steelblue";
        ctx.lineWidth = 10;

        // ctx.globalAlpha = 1;
        let highlightData = data.filter(d=>d.file_id === highlightImage);
        render(highlightData);
    }, [ctx, canvas, highlightImage])

    let canvasStyles = {
        width: innerWidth + "px",
        height: innerHeight + "px",
        marginTop: margin.top,
        marginLeft: margin.left
    }
    let visWrapperStyle = {
        width: width + "px",
        height: height + "px",
        // position: "relative",
        // left: 0,
        // top: 30
    }
    let containerStyle = {
        display: "flex",
        flexDirection: "column",
        marginRight: "40px"
    }

    let headerStyle = {
        marginLeft: "45px",
            marginTop: "10px",
            marginBottom: "5px",
        textAlign: "left"

        // fontSize: 14,
    };

    let headerContainerStyle = {
        background: "#F1F1F1"
    }
    return (
        <Card style={containerStyle}>
            <div style={headerContainerStyle} >
                <Typography style={headerStyle} variant="h5" component="h4">
                    Parallel Coordinates
                </Typography>
            </div>

            <div className={"parcoords"} width={width + margin.left + margin.right}
                 style = {visWrapperStyle}
                 height={height + margin.top + margin.bottom}   >

                <canvas id={"parcoords-canvas"}
                        width={innerWidth + "px"}
                        height={innerHeight + "px"}
                        style={canvasStyles}

                >
                </canvas>
                <svg id={"parcoords-svg"}>

                </svg>
            </div>
        </Card>

    )
}
// container.append("canvas")
//         .attr("width", width * devicePixelRatio)
//         .attr("height", height * devicePixelRatio)
//         .style("width", width + "px")
//         .style("height", height + "px")
//         .style("margin-top", margin.top + "px")
//         .style("margin-left", margin.left + "px");
//