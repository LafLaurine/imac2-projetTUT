package com.afp.medialab.weverify.fakedetection;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
public class MesoNetTrainingController {
	@GetMapping("/api/MesoNet/training")
	public String extraction() {
		RestTemplate rTemplate =  new RestTemplate();
		String fooResourceUrl = "http://0.0.0.0:5000/mesonet_training";
		ResponseEntity<String> response = rTemplate.getForEntity(fooResourceUrl, String.class);
        return "MesoNet training done : " + response;
    }
}