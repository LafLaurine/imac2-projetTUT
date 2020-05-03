import { useEffect} from "react";
import axios from "axios"
import {useDispatch} from "react-redux";
import{setFakeDetectionLoading, setFakeDectectionResult} from "../../../../../redux/actions/tools/fakeDetectionActions";
import {setError} from "../../../../../redux/actions/errorActions";
import useLoadLanguage from "../../../../../Hooks/useLoadLanguage";
import tsv from "../../../../../LocalDictionary/components/NavItems/tools/Forensic.tsv";

const useGetAnalyse = (url) => {
    const keyword = useLoadLanguage("components/NavItems/tools/Forensic.tsv", tsv);
    const dispatch = useDispatch();

    useEffect(() => {

        const handleError = (e) => {
            if (keyword(e) !== "")
                dispatch(setError(keyword(e)));
            else
                dispatch(setError(keyword("please_give_a_correct_link")));
            dispatch(setFakeDetectionLoading(false));
        };

        const getResult = () => {
            axios.get("http://localhost:8080/api/MesoNet/analyse")
                .then(response => {
                    if (response.data.status === "completed") {
                        dispatch(setFakeDectectionResult(url, response.data, false, false));
                    } else {
                        handleError("fakedetection_error_" + response.data.status);
                    }
                })
                .catch(error => {
                    handleError("fakedetection_error_" + error.status);
                })
        };

        const waitUntilFinish = () => {
            axios.get("http://localhost:8080/api/MesoNet/analyse")
                .then((response) => {
                    if (response.data.status === "processing") {
                        setTimeout(function () {
                            waitUntilFinish();
                        }, 2000);
                    } else if (response.data.status === "completed") {
                        getResult();
                    } else {
                        handleError("fakedetection_error_" + response.data.status);
                    }
                })
                .catch(error => {
                    handleError("fakedetection_error_" + error.status);
                })
        };


        const newFakeDetectionRequest = (data) => {
            if (data.status === "downloaded")
                waitUntilFinish();
            else if (data.status === "exist")
                getResult();
            else {
                handleError("fakedetection_error_" + data.status);
            }
        };

        if (url) {
            dispatch(setFakeDetectionLoading(true));
            axios.get("http://localhost:8080/api/MesoNet/analyse")
                .then(response => newFakeDetectionRequest(response.data))
                .catch(error => {
                    handleError("fakedetection_error_" + error.status);
                })
        }
    }, [url, keyword]);
};
export default useGetAnalyse;