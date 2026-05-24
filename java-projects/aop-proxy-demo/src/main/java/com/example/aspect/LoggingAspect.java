package com.example.aspect;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class LoggingAspect {

    @Around("execution(* com.example.service.*.*(..))")
    public Object logMethodCall(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().getName();
        System.out.println("Entering method: " + methodName);
        try {
            return joinPoint.proceed();
        } catch (Throwable e) {
            System.out.println("Exception in method: " + methodName);
            throw e;
        } finally {
            System.out.println("Exiting method: " + methodName);
        }
    }
}