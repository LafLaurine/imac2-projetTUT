package com.afp.medialab.weverify.fakedetection;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class CapsuleForensicsAnalyseController {
	@GetMapping("/api/CapsuleForensics/analyse")
	public String capsuleAnalyse() {
		RestTemplate rTemplate =  new RestTemplate();
		String fooResourceUrl = "http://0.0.0.0:5000/capsule_forensics_analyse";
		ResponseEntity<String> response = rTemplate.getForEntity(fooResourceUrl, String.class);
        return "CapsuleForensics analyse done : " + response;
    }
}