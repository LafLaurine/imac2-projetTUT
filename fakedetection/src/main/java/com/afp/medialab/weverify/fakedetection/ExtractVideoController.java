package com.afp.medialab.weverify.fakedetection;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class ExtractVideoController {
	@GetMapping("/api/extraction/video")
	public String extraction() {
		RestTemplate rTemplate =  new RestTemplate();
		String fooResourceUrl = "http://0.0.0.0:5001/extract_faces_video";
		ResponseEntity<String> response = rTemplate.getForEntity(fooResourceUrl, String.class);
        String responseStr = response.getBody();
		int begin = responseStr.indexOf("{");
		int end = responseStr.lastIndexOf("}") + 1;
		responseStr = responseStr.substring(begin, end);
        return responseStr;
    }
}