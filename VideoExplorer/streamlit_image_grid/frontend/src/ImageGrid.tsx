// import {GridList, GridTile} from "material-ui";

import './ImageGrid.css'
import React, {ComponentProps, CSSProperties, useState} from 'react';
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
import List from '@material-ui/core/List';
import ListItem from "@material-ui/core/ListItem";
import ExpanderIcon from '@material-ui/icons/ExpandMore';
import CloseExpanderIcon from '@material-ui/icons/ExpandLess';
interface ImageGridProps {
    onClickPlay: (config: ImageConfig) => void,
    imageConfigs: Array<ImageConfig>,
    videoShowing: boolean,
    columns: Array<Column>,
    imageDataMap: Map<string, ImageData>
}

enum SortingWay {
    NoSorting, High2Low, Low2High,
}

function ImageGrid({onClickPlay, imageConfigs, videoShowing, columns, imageDataMap}: ImageGridProps) {
    let possibleViews = ['A', 'B', 'C', 'D'];
    let height = 800;
    let width = 960;
    let columnNumber = 3;
    let spacing = 80;
    let gridImageWidth = (width - (columnNumber - 1) * spacing) / columnNumber;
    let gridImageHeight = gridImageWidth * 54 / 96;
    if (videoShowing) {
        height = 450
    }
    let expanderWidth = Math.round(width - spacing*(columnNumber-2) )
    const [expandingImage, setExpandingImage] = useState<ImageConfig>();
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
                alignItems: "flex-end",
                flexGrowth: 0,
            },
            headerItem: {
                marginLeft: "5px",
                marginBottom: "3px"
            },
            gridContainer: {
                width: "100%",
                flexGrowth: 2,
                overflowY: "scroll",
                height: height + "px",
                paddingTop: "20px"
            },
            gridRow: {
                display: "flex",
                flexDirection: "row",
                justifyContent: "space-evenly",
                marginBottom: "15px"
            },
            gridExpander: {
                display: "flex",
                flexDirection: "row",
                justifyContent: "center",
                // justifyContent: "space-evenly",
                marginRight: "10px",
                marginLeft: "10px",
                paddingTop: "13px",
                paddingBottom: "14px",
                marginBottom: "4px",
                background: "#F1F1F1",
                width: expanderWidth + "px",

            },
            gridImageContainer: {
                // background: "black",
                // opacity: 0.4,
                // background: "#2877cc",
                borderRadius: "10px",
                paddingTop: "5px",
                paddingLeft: "8px",
                paddingRight: "8px",
                paddingBottom: "5px",
            },
            gridImageItem: {
                // marginRight: spacing + "px",
                width: gridImageWidth + "px",

                position: "relative",
                left: 0,
                top: 0
            },
            gridImageExpander: {
                display: "flex",
                flexDirection: "row",
                justifyContent: "space-evenly",
                marginBottom: "4px",
                background: "steelblue"
            },
            gridImageBar: {
                background: "black",
                opacity: 0.4,
                width: "100%",
                height: Math.round(gridImageHeight * 0.4) + "px",
                position: "absolute",
                left: 0,
                bottom: 0,
                // top : Math.round(gridImageHeight * 0.4 + 3 - 20) + "px" ,
                // marginBottom: Math.round(gridImageHeight * 0.4 + 3) + "px",
                display: "flex",
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "space-between"
            },
            gridImageText: {
                opacity: 1,
                color: "white",

            },
            gridImageTitle: {},
            gridImageInfo: {
                fontSize: 10
            },
            gridExpanderContainer : {
                position: "relative",
                left: 0,
                top: 0,
                height: Math.round(gridImageHeight * 1.4) + "px"

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

    let renderColumnSelect = (col: Column) => {
        return (<MenuItem value={col.key}>{descriptionMap.get(col.key)}</MenuItem>)
    }
    let defaultColumn = "";
    if (columns.length > 0) {
        defaultColumn = columns[0].key;
    }
    let realShowingMetric = metricShowing ? metricShowing : defaultColumn;
    if (realShowingMetric) {

    }

    let sortedImages = imageConfigs;
    let extractValue = (config: ImageConfig) => {
        let tmp = imageDataMap.get(config.file_id)
        if (tmp) {
            let r = tmp.data.get(realShowingMetric)
            if (r) {
                return r;
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }
    if (sorting === SortingWay.High2Low) {
        sortedImages = imageConfigs.sort((a, b) => {
            return -extractValue(a) + extractValue(b);
        })
    } else if (sorting === SortingWay.Low2High) {
        sortedImages = imageConfigs.sort((a, b) => {
            return extractValue(a) - extractValue(b);
        })
    }
    let handleSwitchChange = (sortingWay: SortingWay) => {
        if (sortingWay === sorting) {
            setSorting(SortingWay.NoSorting);
        } else {
            setSorting(sortingWay);
        }
        // setSorting(SortingWay.High2Low);
    }
    let onClickExpander = (imageConfig: ImageConfig) => {
        if(imageConfig.file_id === expandingImage?.file_id){
            setExpandingImage(undefined);
        }else{
            setExpandingImage(imageConfig);
        }
    }
    let renderImage = (tile: ImageConfig, i: number) => {
        let {title} = describeImage(tile);
        let fileId = tile.file_id;
        let tmp = imageDataMap.get(fileId);
        let realValue = 0
        if (!!tmp) {
            let value = tmp.data.get(realShowingMetric);
            if (value) {
                realValue = value;
            }
        }
        let subtitle = ""
        if (columns.length > 0) {
            subtitle = `${descriptionMap.get(realShowingMetric)}: ${realShowingMetric ? Math.round(realValue * 100) / 100 : 0}`;
        }
        let renderExpandIcon = (image: ImageConfig,) => {
            if(image.file_id === expandingImage?.file_id){
                return (
                    <CloseExpanderIcon onClick={() => onClickExpander(image)}/>
                )
            }else if(image.play === expandingImage?.play){
                return
            }else{
                return <ExpanderIcon onClick={() => onClickExpander(image)}/>
            }
        }
        return (
            <div className={classes.gridImageContainer}>
            <div className={classes.gridImageItem}>
                <img width={gridImageWidth + "px"} src={"/video_images/" + tile.file_id + ".jpg"} alt={""}/>
                <div className={classes.gridImageBar}>
                    <div></div>
                    <div className={classes.gridImageText}>
                        <span className={classes.gridImageTitle}> {title} </span>
                        <br/>
                        <span className={classes.gridImageInfo}> {subtitle} </span>
                    </div>
                    <div>

                        <IconButton aria-label={`info about ${describeImage(tile).title}`}
                                    className={classes.icon}>
                            <PlayerIcon onClick={() => onClickPlay(tile)}/>
                        </IconButton>
                        <IconButton aria-label={`info about ${describeImage(tile).title}`}
                                    className={classes.icon}>
                            {renderExpandIcon(tile)}
                            {/*<ExpanderIcon onClick={() => onClickExpander(tile)}/>*/}
                        </IconButton>
                    </div>

                </div>
                {/*<GridListTileBar*/}
                {/*    title={describeImage(tile).title}*/}
                {/*    subtitle={<span>{subtitle}</span>}*/}
                {/*    actionIcon={*/}

                {/*    }*/}
                {/*/>*/}
            </div>
            </div>
        )
    }
    let renderRow = (images: Array<ImageConfig>, rowId: number) => {
        return (
            <div className={classes.gridRow}>
                {images.map((tile, i) => renderImage(tile, i))}
            </div>
        )
    }
    let renderImageExpander = (image: ImageConfig, col: number) => {
        let expanderImages = []
        console.log(image.file_id);
        for (let view of possibleViews) {
            if (view === image.view) {

            } else {

                let config = {
                    file_id: image.play + "_" + view,
                    game: image.game,
                    view: view,
                    play: image.play,
                    col: image.col
                }

                expanderImages.push(config);
            }
        }
        let i = 3;
        let expanderArrowStyle: CSSProperties = {
            position: "absolute",
                top: 0,
                left:  (width - columnNumber * gridImageWidth - (columnNumber - 4)* spacing ) + 65  + width * col/columnNumber
        }
        let expanderStyle: CSSProperties = {
            position: "relative",
            top: 0,
            left: (width - columnNumber * gridImageWidth - (columnNumber - 1)* spacing ) +65  + (width - expanderWidth )* col/columnNumber,
            // marginLeft: (col * 10 ) + "px",
            // marginRight: (Math.round(0.6 * width - col*10)) + "px"
        }
        let arrowStyle
        if(col === columnNumber - 1){
            expanderStyle = {
                position: "absolute",
                top: 0,
                right: width - columnNumber * gridImageWidth - spacing  ,
            }
        }


        return (
            <div className={classes.gridExpanderContainer}>
                <div id={"expanderArrow"} style={expanderArrowStyle}></div>
                <Card className={classes.gridExpander} style={expanderStyle}>
                    {expanderImages.map((tile, i) => renderImage(tile, i))}
                </Card>
            </div>

        )
    }
    let renderImages = () => {
        let currentRow = [];
        let renderedRows = [];
        let expandingCol = -1;
        for (let image of sortedImages.slice(0, 30)) {
            if (image.file_id === expandingImage?.file_id) {
                expandingCol = currentRow.length
            }
            currentRow.push(image);
            if (currentRow.length === 3) {
                renderedRows.push(renderRow(currentRow, 0));
                if (expandingCol >= 0 && expandingImage) {
                    renderedRows.push(renderImageExpander(expandingImage, expandingCol));
                }
                currentRow = [];
                expandingCol = -1;
            } else {
            }
        }
        return renderedRows.map(d => d);
        // return sortedImages.slice(0, 30).map((tile, i) => {
        //     return renderImage(tile, i);
        // });
    }
    return (
        <Card className={classes.root}>
            <div className={classes.header}>
                <FormControl className={classes.formControl}>
                    <InputLabel id="demo-simple-select-label">Metric</InputLabel>
                    <Select
                        labelId="demo-simple-select-label"
                        id="demo-simple-select"
                        value={metricShowing ? metricShowing : defaultColumn}
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
                                          checked={sorting === SortingWay.High2Low}
                                          onChange={() => handleSwitchChange(SortingWay.High2Low)}
                                          name="high2low"
                                          color="primary"
                                      />
                                  }
                                  label="High"
                />
                <FormControlLabel className={classes.headerItem}
                                  control={
                                      <Switch
                                          checked={sorting === SortingWay.Low2High}
                                          onChange={() => handleSwitchChange(SortingWay.Low2High)}
                                          name="low2high"
                                          color="primary"
                                      />
                                  }
                                  label="Low"
                />
            </div>
            <div className={classes.gridContainer}>
                {/*<List>*/}
                {
                    renderImages()
                }
                {/*</List>*/}


            </div>
            {/*<GridList cellHeight={180} spacing={30} className={classes.gridList}>*/}
            {/*    <GridListTile key="Subheader" cols={2} style={{height: 'auto'}}>*/}
            {/*        <ListSubheader component="div"></ListSubheader>*/}
            {/*    </GridListTile>*/}
            {/*    {sortedImages.slice(0, 30).map((tile, i) => {*/}
            {/*        let {title} = describeImage(tile);*/}
            {/*        let fileId = tile.file_id;*/}
            {/*        let tmp = imageDataMap.get(fileId);*/}
            {/*        let realValue = 0*/}
            {/*        if(!!tmp){*/}
            {/*            let value = tmp.data.get(realShowingMetric);*/}
            {/*            if(value){*/}
            {/*                realValue = value;*/}
            {/*            }*/}
            {/*        }*/}
            {/*        let subtitle = ""*/}
            {/*        if(columns.length > 0){*/}
            {/*            subtitle = `${descriptionMap.get(realShowingMetric)}: ${realShowingMetric?Math.round(realValue*100)/100: 0}`;*/}
            {/*        }*/}
            {/*        return (*/}
            {/*            <GridListTile key={tile.file_id} cols={0.65}>*/}
            {/*                <img src={"/video_images/" + tile.file_id + ".jpg"} alt={title}/>*/}
            {/*                <GridListTileBar*/}
            {/*                    title={describeImage(tile).title}*/}
            {/*                    subtitle={<span>{subtitle}</span>}*/}
            {/*                    actionIcon={*/}
            {/*                        <IconButton aria-label={`info about ${describeImage(tile).title}`}*/}
            {/*                                    className={classes.icon}>*/}
            {/*                            <PlayerIcon onClick={()=>onClickPlay(tile)} />*/}
            {/*                        </IconButton>*/}
            {/*                    }*/}
            {/*                />*/}
            {/*            </GridListTile>*/}
            {/*        )*/}
            {/*    })}*/}
            {/*</GridList>*/}
        </Card>
    )

}

export default ImageGrid;