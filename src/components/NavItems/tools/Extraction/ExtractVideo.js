import React, {useState} from "react";
import Button from "@material-ui/core/Button";
import useExtraction from "./Hooks/useExtraction"
import useLoadLanguage from "../../../../Hooks/useLoadLanguage";
import tsv from "../../../../LocalDictionary/components/NavItems/tools/Extraction.tsv";

const ExtractVideo = () => {
    const [{ data, isLoading, isError }] = useExtraction("http://localhost:8080/api/extraction/video",{ extraction: [] },);
    const keyword = useLoadLanguage("components/NavItems/tools/Extraction.tsv", tsv);
    const [clicked, setClicked] = useState(false);
    
    function doClick()  {
        setClicked(true);
    }

    return (<div> 
        {isError && <div>Something went wrong with {keyword("extraction_video")} ...</div>}
        {isLoading ? (
        <div>Loading {keyword("extraction_video")} ...</div>
        ) : (
            <div>
                <Button variant="contained" color="primary" onClick={clicked ? undefined : doClick}>{keyword("extraction_video")}</Button>
                {clicked && <h3>{data.message}</h3>}
            </div>
            )}
        </div>
    ); 
}

export default ExtractVideo;