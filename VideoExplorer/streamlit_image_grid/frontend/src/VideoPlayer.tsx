import React, {CSSProperties, useRef} from "react";
import ReactPlayer from 'react-player/file'
import Card from '@material-ui/core/Card'
// import  from '@material-ui/icons/Close';
import {Close} from "@material-ui/icons";
import {describeColumns, describeImage} from "./utils";
import CloseIcon from "@material-ui/icons/Close";
import IconButton from "@material-ui/core/IconButton";
import {createStyles, makeStyles, Theme} from "@material-ui/core/styles";
interface VideoPlayerPros {
    url: string,
    onClose: () => void
}


function VideoPlayer({url, onClose}: VideoPlayerPros) {
    console.log(url);
    const useStyles = makeStyles((theme: Theme) =>  createStyles({

        icon: {
            color: 'rgba(55, 55, 55, 0.54)',
            position: "absolute",
            top: 15,
            right: 20,
            width: 60,
            height: 60
        },

    }));
    const classes = useStyles();
    // const sourceRef = useRef<HTMLHeadingElement>(null)
    const playerStyles: CSSProperties = {
        position: "relative",
        bottom: 0,

        // left: 0,

    }
    const containerStyle:CSSProperties = {
        width: "100%",
        height: "400px",
        border: "steelblue",
        background: "#fafafa",
        display: "block",
        marginLeft: "18px",
        marginTop: "10px",
        marginRight: "1px",
        marginBottom: "10px"
    }
    let closeStyle: CSSProperties = {

    }
    let onClickClose = ()=>{
        onClose();
    }
    return (
        <Card style={containerStyle}>
            <ReactPlayer
                style={playerStyles}
                width={"100%"}
                height={"400px"}
                url={url}
                playing={true}
                controls={true}
            ></ReactPlayer>
            {/*<video width="480" height="270" controls>*/}
            {/*    <source*/}
            {/*        // ref={sourceRef}*/}
            {/*        src={url}*/}
            {/*        // src="https://storage.googleapis.com/ltpt-videos/AFL%20Video/181101_AFL-PEORIA_AFL-SCOTTSDALE__0/3367B.mp4"*/}
            {/*        // src="/test.mp4"*/}
            {/*        type="video/mp4"*/}
            {/*    />*/}
            {/*</video>*/}
            <IconButton aria-label={`close`}
                        className={classes.icon}>
                <CloseIcon onClick={()=>onClickClose()} />
            </IconButton>
        </Card>
    )
}

export default VideoPlayer