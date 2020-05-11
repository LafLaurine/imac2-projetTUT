import React, {useState} from "react";
import Button from "@material-ui/core/Button";
import useTraining from "./Hooks/useTraining";
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/FakeDetectionTraining.tsv";

const FakeDetectionTrainingCapsule = () => {
    const [{ data, isLoading, isError }] = useTraining("http://localhost:8080/api/CapsuleForensics/training",{ training: [] },);
    const keyword = useLoadLanguage("components/NavItems/tools/FakeDetectionTraining.tsv", tsv);
    const [clicked, setClicked] = useState(false);
   
    function doClick() {
        setClicked(true);
    }

    return (<div> 
        {isError && <div>Something went wrong with CapsuleForensics training ...</div>}
        {isLoading ? (
        <div>Loading {keyword("fakedetectiontraining_trainc")} ...</div>
        ) : (
            <div>
                <h1> {keyword("fakedetectiontraining_trainc")} </h1>
                <Button variant="contained" color="primary" onClick={clicked ? undefined : doClick}>{keyword("fakedetectiontraining_trainc")}</Button>
                {clicked &&
                <div>
                <h3>Evaluation training accuracy : [{data._EvaluationLearning__acc_training && data._EvaluationLearning__acc_training.join(', ')}]</h3>
                <h3>Evaluation validation accuracy : [{data._EvaluationLearning__acc_validation && data._EvaluationLearning__acc_validation.join(', ')}]</h3>
                <h3>Evaluation epochs : [{data._EvaluationLearning__epochs && data._EvaluationLearning__epochs.join(', ')}]</h3>
                <h3>Evaluation training loss : [{data._EvaluationLearning__loss_training && data._EvaluationLearning__loss_training.join(', ')}]</h3>
                <h3>Evaluation validation loss : [{data._EvaluationLearning__loss_validation && data._EvaluationLearning__loss_validation.join(', ')}]</h3>
                </div>
                }
            </div>
            )}
        </div>
    ); 
}

export default FakeDetectionTrainingCapsule;