// import {GridList, GridTile} from "material-ui";
import React, {ComponentProps, useState} from 'react';
import {Theme, createStyles, makeStyles} from '@material-ui/core/styles';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';
import ListSubheader from '@material-ui/core/ListSubheader';
import IconButton from '@material-ui/core/IconButton';
import PlayerIcon from '@material-ui/icons/PlayCircleOutline';
import FormControl from '@material-ui/core/FormControl'
import {ImageConfig, ImageData, Column, describeImage, describeColumns} from "./utils";
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormHelperText from '@material-ui/core/FormHelperText';
import Select from '@material-ui/core/Select';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
import Paper from "@material-ui/core/Paper";
import Card from '@material-ui/core/Card'



interface ImageGridProps {
    onClickPlay: (config: ImageConfig)=>void,
    imageConfigs: Array<ImageConfig>,
    videoShowing: boolean,
    columns: Array<Column>,
    imageDataMap: Map<string, ImageData>
}
enum SortingWay {
    NoSorting, High2Low,Low2High,
}

function ImageGrid({onClickPlay, imageConfigs, videoShowing, columns, imageDataMap}: ImageGridProps) {
    let height = 800;
    if(videoShowing){
        height = 450
    }
    const useStyles = makeStyles((theme: Theme) =>
        createStyles({
            root: {
                display: 'flex',
                flexWrap: 'wrap',
                justifyContent: 'space-around',
                overflow: 'hidden',
                backgroundColor: theme.palette.background.paper,
                // paddingTop: "10px",
                paddingBottom: "20px",
                // paddingLeft: "1px",
                // paddingRight: "1px",
                marginLeft: "10px",
                marginTop: "10px",
                marginRight: "1px",
                marginBottom: "10px"
            },
            gridList: {
                width: 960,
                height: height,
            },
            icon: {
                color: 'rgba(255, 255, 255, 0.54)',
            },
            formControl: {
                margin: theme.spacing(1),
                minWidth: 120,
            },
            selectEmpty: {
                marginTop: theme.spacing(2),
            },
            header: {
                display: "flex",
                flexDirection: "row",
                justifyContent: "flex-start",
                width: "100%",
                background: "#F1F1F1",
                paddingLeft: "28px",
                paddingBottom: "4px",
                alignItems: "flex-end"
            },
            headerItem: {
                marginLeft: "5px",
                marginBottom: "3px"
            }

        }),
    );
    let {descriptionMap} = describeColumns(columns)
    const classes = useStyles();
    const [metricShowing, setMetricShowing] = useState<string>();
    const [sorting, setSorting] = useState<SortingWay>()
    const handleMetricChange = (event: React.ChangeEvent<{ value: unknown }>) => {
        setMetricShowing(event.target.value as string);
    };
    console.log(columns)

    let renderColumnSelect = (col: Column) => {
        return (<MenuItem value={col.key}>{descriptionMap.get(col.key)}</MenuItem>)
    }
    let defaultColumn = "";
    if(columns.length > 0){
        defaultColumn = columns[0].key;
    }
    let realShowingMetric = metricShowing?metricShowing:defaultColumn;
    if (realShowingMetric){

    }

    let sortedImages = imageConfigs;
    let extractValue = (config: ImageConfig) => {
        let tmp = imageDataMap.get(config.file_id)
        if(tmp){
            let r= tmp.data.get(realShowingMetric)
            if(r){
                return r;
            }else{
                return 0;
            }
        }else{
            return 0;
        }
    }
    if(sorting === SortingWay.High2Low){
        sortedImages = imageConfigs.sort((a, b)=>{
            return -extractValue(a) +extractValue(b);
        })
    }else if(sorting === SortingWay.Low2High){
        sortedImages = imageConfigs.sort((a, b)=>{
            return extractValue(a) - extractValue(b);
        })
    }
    let handleSwitchChange = (sortingWay: SortingWay)=>{
        if(sortingWay === sorting){
            setSorting(SortingWay.NoSorting);
        }else{
            setSorting(sortingWay);
        }
        // setSorting(SortingWay.High2Low);
    }

    return (
        <Card className={classes.root}>
            <div className={classes.header}>
                <FormControl className={classes.formControl}>
                    <InputLabel id="demo-simple-select-label">Metric</InputLabel>
                    <Select
                        labelId="demo-simple-select-label"
                        id="demo-simple-select"
                        value={metricShowing?metricShowing:defaultColumn}
                        onChange={handleMetricChange}
                    >
                        {
                            columns.map(renderColumnSelect)
                        }
                    </Select>
                </FormControl>
                <FormControlLabel className={classes.headerItem}
                    control={
                        <Switch
                            checked={sorting===SortingWay.High2Low}
                            onChange={()=>handleSwitchChange(SortingWay.High2Low)}
                            name="high2low"
                            color="primary"
                        />
                    }
                    label="High"
                />
                <FormControlLabel className={classes.headerItem}
                    control={
                        <Switch
                            checked={sorting===SortingWay.Low2High}
                            onChange={()=>handleSwitchChange(SortingWay.Low2High)}
                            name="low2high"
                            color="primary"
                        />
                    }
                    label="Low"
                />
            </div>

            <GridList cellHeight={180} spacing={30} className={classes.gridList}>
                <GridListTile key="Subheader" cols={2} style={{height: 'auto'}}>
                    <ListSubheader component="div"></ListSubheader>
                </GridListTile>
                {sortedImages.slice(0, 30).map((tile, i) => {
                    let {title} = describeImage(tile);
                    let fileId = tile.file_id;
                    let tmp = imageDataMap.get(fileId);
                    let realValue = 0
                    if(!!tmp){
                        let value = tmp.data.get(realShowingMetric);
                        if(value){
                            realValue = value;
                        }
                    }
                    let subtitle = ""
                    if(columns.length > 0){
                        subtitle = `${descriptionMap.get(realShowingMetric)}: ${realShowingMetric?Math.round(realValue*100)/100: 0}`;
                    }
                    return (
                        <GridListTile key={tile.file_id} cols={0.65}>
                            <img src={"/video_images/" + tile.file_id + ".jpg"} alt={title}/>
                            <GridListTileBar
                                title={describeImage(tile).title}
                                subtitle={<span>{subtitle}</span>}
                                actionIcon={
                                    <IconButton aria-label={`info about ${describeImage(tile).title}`}
                                                className={classes.icon}>
                                        <PlayerIcon onClick={()=>onClickPlay(tile)} />
                                    </IconButton>
                                }
                            />
                        </GridListTile>
                    )
                })}
            </GridList>
        </Card>
    )

}

export default ImageGrid;