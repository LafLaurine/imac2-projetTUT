import {Paper} from "@material-ui/core";
import CustomTile from "../../../Shared/CustomTitle/CustomTitle";
import Box from "@material-ui/core/Box";
import React from "react";
import FakeDetectionTrainingMesoNet from "./FakeDetectionTrainingMesoNet"
import FakeDetectionTrainingCapsule from './FakeDetectionTrainingCapsule'
import useMyStyles from "../../../Shared/MaterialUiStyles/useMyStyles";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/FakeDetectionTraining.tsv";

const FakeDetection = () => {
    const classes = useMyStyles();
    const keyword = useLoadLanguage("components/NavItems/tools/FakeDetectionTraining.tsv", tsv);

    return (
        <div>
            <Paper className={classes.root}>
            <CustomTile text={keyword("fakedetectiontraining_title")}/>
            <Box m={1}/>
            <FakeDetectionTrainingMesoNet></FakeDetectionTrainingMesoNet>
            <FakeDetectionTrainingCapsule></FakeDetectionTrainingCapsule>
            </Paper>
        </div>
 
    );
};
export default FakeDetection;