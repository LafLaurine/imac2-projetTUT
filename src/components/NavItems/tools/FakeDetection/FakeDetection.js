import {Paper} from "@material-ui/core";
import CustomTile from "../../../Shared/CustomTitle/CustomTitle";
import Box from "@material-ui/core/Box";
import FakeDectetionMesoNetResult from "./FakeDetectionMesoNetResult"
import FakeDetectionCapsuleResult from "./FakeDetectionCapsuleResult"

import React from "react";
import useMyStyles from "../../../Shared/MaterialUiStyles/useMyStyles";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/FakeDetection.tsv";

const FakeDetection = () => {
    const classes = useMyStyles();
    const keyword = useLoadLanguage("components/NavItems/tools/FakeDetection.tsv", tsv);

    return (
        <div>
            <Paper className={classes.root}>
            <CustomTile text={keyword("fakedetection_title")}/>
            <Box m={1}/>
            <FakeDectetionMesoNetResult></FakeDectetionMesoNetResult>
            <FakeDetectionCapsuleResult></FakeDetectionCapsuleResult>
            </Paper>
        </div>
 
    );
};
export default FakeDetection;