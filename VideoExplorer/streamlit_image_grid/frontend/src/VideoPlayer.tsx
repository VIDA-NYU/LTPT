import {CSSProperties, useRef} from "react";
import ReactPlayer from 'react-player/file'

interface VideoPlayerPros {
    url: string
}

function VideoPlayer({url}: VideoPlayerPros) {
    console.log(url);
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
        display: "block"
    }
    return (
        <div style={containerStyle}>
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

        </div>
    )
}

export default VideoPlayer