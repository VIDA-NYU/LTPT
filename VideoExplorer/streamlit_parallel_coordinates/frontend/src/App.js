import logo from './logo.svg';
import './App.css';
import ParallelCoordinates from "./ParallelCoordinates.js";
import BarDistribution from "./BarDistribution";
import {generateFakeData} from "./data";
import {setState, useEffect, useState} from "react";
import {Button} from "@material-ui/core"
import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
import {describeColumns} from "./utils";

// let {data, dimensions} = generateFakeData();
function App({args}) {
    const [columnFocus, setColumnFocus] = useState(null);
    const [imageFilter, setImageFilter] = useState([]);
    const [columnFilters, setColumnFilters] = useState([]);
    const [highlightImage, setHighlightImage]= useState(null);
    const [compRect, setCompRect] = useState({
        width: 1000,
        height: 200
    })
    let data = args['data'];
    let columns = args['columns'];
    let onSetImageFilter = (imageIds, activeFilters)=>{
        setImageFilter(imageIds);
        setColumnFilters(activeFilters);
        Streamlit.setComponentValue({
            filtered: imageIds
        })

    }
    let {descriptions} = describeColumns(columns)
    useEffect(()=>{
        Streamlit.setFrameHeight();
        // Streamlit.setFrameWidth();
    }, []);
    let barColumn = columnFocus?columns.filter( d=>d.key ===columnFocus)[0]:columns[0]
    return (
        <div className="App" width={compRect.width}>

                <ParallelCoordinates className="parcoords-container" data={data} dimensions={columns}
                                     setColumnFocus={setColumnFocus} onSetImageFilter={onSetImageFilter}
                                     highlightImage={highlightImage} height={compRect.height} width={compRect.width * 0.55}
                                     columnDescriptions={descriptions}
                >
                </ParallelCoordinates>
                <BarDistribution className="distribution" data={data} column={barColumn}
                                height={compRect.height} width={compRect * 0.4} imageFilter={imageFilter} columnFilters={columnFilters}
                                 columnDescription={descriptions.filter(d=>d.key === barColumn.key)[0]}

                >

                </BarDistribution>
                {/*<Button onClick={()=>{*/}
                {/*    setHighlightImage("1");*/}
                {/*}}>HELLO</Button>*/}
        </div>
    );
}

export default withStreamlitConnection(App);
