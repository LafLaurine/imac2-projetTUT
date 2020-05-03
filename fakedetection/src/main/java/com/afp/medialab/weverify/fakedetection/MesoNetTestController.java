package com.afp.medialab.weverify.fakedetection;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class MesoNetTestController {
	@GetMapping("/api/MesoNet/test")
	public String test() {
		RestTemplate rTemplate =  new RestTemplate();
		String fooResourceUrl = "http://0.0.0.0:5000/mesonet_test";
		ResponseEntity<String> response = rTemplate.getForEntity(fooResourceUrl, String.class);
        return "MesoNet test done : " + response;
    }
}