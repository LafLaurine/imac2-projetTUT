package com.afp.medialab.weverify.fakedetection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.web.client.RestTemplate;


@SpringBootApplication
public class FakedetectionApplication {

	private static final Logger log = LoggerFactory.getLogger(ConsumingRestApplication.class);
	public static void main(String[] args) {
		SpringApplication.run(FakedetectionApplication.class, args);
	}

	@Bean
	public RestTemplate restTemplate() {
		return new RestTemplate();
	}	

}
