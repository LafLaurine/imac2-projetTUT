import {Paper} from "@material-ui/core";
import CustomTile from "../../../Shared/CustomTitle/CustomTitle";
import Box from "@material-ui/core/Box";
import useGetAnalyse from "./Hooks/useGetAnalyse";
import React, {useEffect, useState} from "react";
import useMyStyles from "../../../Shared/MaterialUiStyles/useMyStyles";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/Forensic.tsv";

const FakeDetection = () => {
    const classes = useMyStyles();
    const keyword = useLoadLanguage("components/NavItems/tools/Forensic.tsv", tsv);
    const [{ data, isLoading, isError }, doFetch] = useGetAnalyse("http://localhost:8080/api/MesoNet/analyse",{ analyse: [] },);

    return (
        <div>
            <Paper className={classes.root}>
            <CustomTile text={keyword("forensic_title")}/>
            <Box m={1}/>
            {isError && <div>Something went wrong ...</div>}
            {isLoading ? (
            <div>Loading ...</div>
            ) : (
                    <h1>{data.analyse._Prediction__dict_prop_analysis && Object.values(data.analyse._Prediction__dict_prop_analysis)}</h1>            
                )}
            </Paper>
        </div>
 
    );
};
export default FakeDetection;