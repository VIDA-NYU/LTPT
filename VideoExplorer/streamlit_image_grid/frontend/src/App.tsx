import React, {useEffect, useState} from 'react';
import logo from './logo.svg';
import './App.css';
import ImageGrid from "./ImageGrid";
import VideoPlayer from "./VideoPlayer";
import {ImageConfig, getVideoUrl, ImageData} from "./utils";
import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
function App({args} : ComponentProps) {
    const [focusImage, setFocusImage] = useState<ImageConfig>()
    let playVideo = (config: ImageConfig)=>{
        setFocusImage(config)
    }

    useEffect(()=>{
        Streamlit.setFrameHeight()
    }, [])
    let dataMaps = args.data as Array<Map<string, any>>;
    let imageDataArray: Array<ImageData> = []
    for (let dataMap of args.data){

        let metrics = new Map<string, number>();
        let file = "";
        for (let key in dataMap){
            if(key === "file_id"){
                file= dataMap[key];
            }else{
                metrics.set(key, dataMap[key]);
            }
        }
        let imageData: ImageData = {
            file: file,
            data: metrics
        }
        imageDataArray.push(imageData);
    }
    let imageDataMap = new Map<string, ImageData>();
    for (let imageData of imageDataArray){
        imageDataMap.set(imageData.file, imageData);
    }
    let imageConfigs = args.images;
    // for (let i = 0; i < 10; i++) {
    //     imageConfigs.push({
    //         "file_id": "3367_C",
    //         "view": "C",
    //         "play": "3367",
    //         "game": "181101_AFL-PEORIA_AFL-SCOTTSDALE__0",
    //         "col": "AFL Video"
    //     })
    // }
    let onVideoClose = ()=>{
        setFocusImage(undefined);
    }
    let url = getVideoUrl(focusImage);
    let renderVideo = ()=>{
        if(focusImage!==undefined){
            return ( <VideoPlayer onClose={onVideoClose} url={url}/>)
        }else{
            // return (<VideoPlayer url={url}/>)
        }
    }

    return (
        <div className="App">

            {
                renderVideo()
            }
            <ImageGrid imageConfigs={imageConfigs} onClickPlay={playVideo} videoShowing={!focusImage? false: true}
                columns={args.columns} imageDataMap={imageDataMap}
            />
        </div>
    );
}

export default withStreamlitConnection(App);
