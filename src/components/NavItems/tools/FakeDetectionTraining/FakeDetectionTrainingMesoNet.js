import React, {useState} from "react";
import Button from "@material-ui/core/Button";
import useTraining from "./Hooks/useTraining";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/FakeDetectionTraining.tsv";

const FakeDetectionTrainingMesoNet = () => {
    const [{ data, isLoading, isError }] = useTraining("http://localhost:8080/api/MesoNet/training",{ training: [] },);
    const keyword = useLoadLanguage("components/NavItems/tools/FakeDetectionTraining.tsv", tsv);
    const [clicked, setClicked] = useState(false);
   
    function doClick() {
        setClicked(true);
    }

    return (
    <div>
        {isError && <div>Something went wrong with MesoNet training...</div>}
        {isLoading ? (
        <div>Loading {keyword("fakedetectiontraining_trainm")}...</div>
        ) : (
            <div>
                <h1> {keyword("fakedetectiontraining_trainm")} : </h1>
                <Button variant="contained" color="primary" onClick={clicked ? undefined : doClick}>{keyword("fakedetectiontraining_trainm")}</Button>
                <p>{clicked && data && Object.values(data).slice(1,-1)}</p>
            </div>
            )}
    </div>
    ); 
}

export default FakeDetectionTrainingMesoNet;