import {Paper} from "@material-ui/core";
import axios from "axios"
import {setError} from "../../../../redux/actions/errorActions";
import CustomTile from "../../../Shared/CustomTitle/CustomTitle";
import Box from "@material-ui/core/Box";
import Button from "@material-ui/core/Button";
import React, {useEffect, useState} from "react";
import useMyStyles from "../../../Shared/MaterialUiStyles/useMyStyles";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/Forensic.tsv";

const FakeDetection = () => {
    const classes = useMyStyles();
    const keyword = useLoadLanguage("components/NavItems/tools/Forensic.tsv", tsv);

    const getResult = () => {
        axios.get("http://localhost:8080/api/MesoNet/analyse")
            .then(response => {
                return response.data;
            })
            .catch(error => {
               return error.status;
            })
    };

    return (
        <div>
            <Paper className={classes.root}>
            <CustomTile text={keyword("forensic_title")}/>
            <Box m={1}/>
            <Button variant="contained" color="primary">{keyword("button_localfile")}</Button>
            {console.log(getResult())}
            <div><h1>{getResult()}</h1></div>        
            </Paper>
        </div>
    );
};
export default FakeDetection;