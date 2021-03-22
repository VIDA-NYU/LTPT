// import {GridList, GridTile} from "material-ui";
import React, {ComponentProps} from 'react';
import {Theme, createStyles, makeStyles} from '@material-ui/core/styles';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';
import ListSubheader from '@material-ui/core/ListSubheader';
import IconButton from '@material-ui/core/IconButton';
import PlayerIcon from '@material-ui/icons/PlayCircleOutline';

import {ImageConfig, describeImage} from "./utils";

const useStyles = makeStyles((theme: Theme) =>
    createStyles({
        root: {
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-around',
            overflow: 'hidden',
            backgroundColor: theme.palette.background.paper,
        },
        gridList: {
            width: 960,
            height: 450,
        },
        icon: {
            color: 'rgba(255, 255, 255, 0.54)',
        },
    }),
);



interface ImageGridProps {
    onClickPlay: (config: ImageConfig)=>void,
    imageConfigs: Array<ImageConfig>
}
function ImageGrid({onClickPlay, imageConfigs}: ImageGridProps) {
    const classes = useStyles();
    return (
        <div className={classes.root}>
            <GridList cellHeight={180} spacing={30} className={classes.gridList}>
                <GridListTile key="Subheader" cols={2} style={{height: 'auto'}}>
                    <ListSubheader component="div"></ListSubheader>
                </GridListTile>
                {imageConfigs.slice(0, 30).map((tile, i) => {
                    let {title, subtitle} = describeImage(tile)
                    return (
                        <GridListTile key={tile.file_id} cols={0.65}>
                            <img src={"/video_images/" + tile.file_id + ".jpg"} alt={title}/>
                            <GridListTileBar
                                title={describeImage(tile).title}
                                subtitle={<span>by: {describeImage(tile).subtitle}</span>}
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
        </div>
    )

}

export default ImageGrid;