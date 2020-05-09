import {Paper} from "@material-ui/core";
import CustomTile from "../../../Shared/CustomTitle/CustomTitle";
import Box from "@material-ui/core/Box";
import React from "react";
import ExtractVideo from "./ExtractVideo"
import ExtractDirectory from "./ExtractDirectory"
import useMyStyles from "../../../Shared/MaterialUiStyles/useMyStyles";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/Extraction.tsv";

const Extraction = () => {
    const classes = useMyStyles();
    const keyword = useLoadLanguage("components/NavItems/tools/Extraction.tsv", tsv);
    return (
        <div>
            <Paper className={classes.root}>
            <CustomTile text={keyword("extraction_face")}/>
            <Box m={1}/>
            <ExtractVideo></ExtractVideo>
            <ExtractDirectory></ExtractDirectory>
            </Paper>
        </div>
 
    );
};
export default Extraction;