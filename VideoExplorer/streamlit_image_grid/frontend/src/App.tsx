import React, {useEffect, useState} from 'react';
import logo from './logo.svg';
import './App.css';
import ImageGrid from "./ImageGrid";
import VideoPlayer from "./VideoPlayer";
import {ImageConfig, getVideoUrl} from "./utils";
import {withStreamlitConnection, StreamlitComponentBase, Streamlit, ComponentProps} from "streamlit-component-lib"
function App({args} : ComponentProps) {
    const [focusImage, setFocusImage] = useState<ImageConfig>()
    let playVideo = (config: ImageConfig)=>{
        setFocusImage(config)
    }

    useEffect(()=>{
        Streamlit.setFrameHeight()
    }, [])
    console.log(args.images)
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
    let url = getVideoUrl(focusImage);
    return (
        <div className="App">
            <VideoPlayer url={url}/>
            <ImageGrid imageConfigs={imageConfigs} onClickPlay={playVideo}/>
        </div>
    );
}

export default withStreamlitConnection(App);
