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
interface Column {
    name: string,
    key: string,
}

interface ImageData {
    file: string,
    data: Map<string, number>,
}

function describeColumns(columns: Array<Column>){
    let nameCount = columns.reduce(( acc, c)=> {
        if(acc.has(c.name)){
            acc.set(c.name, acc.get(c.name) + 1);
        }else{
            acc.set(c.name, 1);
        }
        return acc;
    },new Map())
    let descriptions = columns.map(col=>{
        if(nameCount.get(col.name) > 1){
            return {
                key: col.key,
                desc: col.name + "-" + col.key
            }
        }else{
            return {
                key: col.key,
                desc: col.name
            };
        }
    })
    let descriptionMap = descriptions.reduce((acc, c)=>{
        acc.set(c.key, c.desc);
        return acc;
    }, new Map())
    return {descriptionMap};
}
export {describeImage, getVideoUrl, describeColumns}
export type { ImageConfig, Column, ImageData }

