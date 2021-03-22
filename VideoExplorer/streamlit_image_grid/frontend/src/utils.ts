interface ImageConfig {
    file_id: string,
    col: string
    game: string,
    play: string,
    view: string,
}

function describeImage(imageConfig: ImageConfig) {
    return {
        title: imageConfig.play + " " + imageConfig.view,
        subtitle: imageConfig.game
    }
}

function getVideoUrl(imageConfig: ImageConfig | undefined){
    if (!imageConfig){
        return "";
    }
    // https://storage.googleapis.com/ltpt-videos/AFL%20Video/181101_AFL-PEORIA_AFL-SCOTTSDALE__0/3367B.mp4
    let raw = `https://storage.googleapis.com/ltpt-videos/${imageConfig.col}/${imageConfig.game}/${imageConfig.play+imageConfig.view}.mp4`
    return raw.replace(" ", "%20")
}

export {describeImage, getVideoUrl}
export type { ImageConfig }

