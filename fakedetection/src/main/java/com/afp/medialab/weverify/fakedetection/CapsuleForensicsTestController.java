package com.afp.medialab.weverify.fakedetection;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class CapsuleForensicsTestController {
	@GetMapping("/api/CapsuleForensics/test")
	public String test() {
		RestTemplate rTemplate =  new RestTemplate();
		String fooResourceUrl = "http://0.0.0.0:5000/capsule_forensics_test";
		ResponseEntity<String> response = rTemplate.getForEntity(fooResourceUrl, String.class);
        return "CapsuleForensics test done : " + response;
    }
}