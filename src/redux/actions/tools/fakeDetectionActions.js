export const setFakeDectectionResult = (url, result, notification, loading) => {
    return {
        type : "SET_FAKEDETECTION_RESULT",
        payload : {
            notification : notification,
            loading : loading,
            url : url,
            result : result,
        }
    }
};

export const setFakeDetectionLoading = (loading) => {
    return {
        type : "SET_FAKEDETECTION_LOADING",
        payload : loading
    }
};

export const cleanFakeDetectionState = () => {
    return {
        type : "FAKEDETECTION_CLEAN_STATE"
    }
};