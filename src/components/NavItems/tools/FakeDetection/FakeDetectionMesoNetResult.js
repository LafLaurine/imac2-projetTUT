import React, {useState} from "react";
import Button from "@material-ui/core/Button";
import useGetAnalyse from "./Hooks/useGetAnalyse";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/FakeDetection.tsv";

const FakeDetectionMesoNetResult = () => {
    const keyword = useLoadLanguage("components/NavItems/tools/FakeDetection.tsv", tsv);
    const [{ data, isLoading, isError }] = useGetAnalyse("http://localhost:8080/api/MesoNet/analyse",{ analyse: [] },);
    const [clicked, setClicked] = useState(false);

    function doClick()  {
        setClicked(true);
    }
 
    return (
    <div>
        {isError && <h3>Something went wrong with {keyword("fakedetection_analysem")}...</h3>}
        {isLoading ? (
        <div>Loading {keyword("fakedetection_analysem")} ...</div>
        ) : (
            <div>
                <h1> {keyword("fakedetection_analysem")} : </h1>
                <Button variant="contained" color="primary" onClick={clicked ? undefined : doClick}> {keyword("fakedetection_analysem")}</Button> 
                <h3>{clicked && data.analyse._Prediction__dict_prop_analysis && Object.values(data.analyse._Prediction__dict_prop_analysis).join(' ')}</h3>
            </div>
            )}
    </div>
    ); 
}

export default FakeDetectionMesoNetResult;