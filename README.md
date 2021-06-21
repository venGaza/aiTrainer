### Exercise Grader with CV

#### Table of Contents
- [Exercise Grader with CV](#exercise-grader-with-cv)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Demo](#demo)
  - [Project Details](#project-details)
    - [Language](#language)
    - [Primary Modules](#primary-modules)
    - [Design Thoughts/Challenges](#design-thoughtschallenges)
    - [Potential Use Cases](#potential-use-cases)
  
#### Project Overview

<p>
Sports serve as an outlet for our competitive nature and at the highest levels, adherence to the rules becomes critical to determining a winner. In most cases, a referee (or many) is designated to supervise the event and enforce the designated ruleset. However, any person who has coordinated a sporting event knows it a challenge to find competent referees to overwatch an event. This occurs because referees typically require some form of experience to provide credibility to their calls, and people would rather enjoy their time playing the sport rather than being on the sidelines enforcing the rules. Can computer vision help fill the gap?
</p>

<p>
CrossFit is one such sporting event that uses a myriad of referees. Athletes test their strength and endurance by completing a variety of fitness challenges. Some of these challenges require repetitions of a movement to complete. In this case, the referee would ensure the athlete is performing the movement correctly before counting a repetition. In this project, I use computer vision to count repetitions of various exercises and see if it could truly replace a human referee. 
</p>



#### Demo

<p align="center"><img src="demo.gif?raw=true" width="60%" height="60%"></p>
  
#### Project Details

##### Language 
- Python

##### Primary Modules
- openCV
- MediaPipe

##### Design Thoughts/Challenges
- Decomposing the grading criterion for exercises was the first challenge to solve. Let's take the squat exercise for example. A person bends their knees until the upper leg (quadriceps/hamstrings) are parallel to the floor. This creates a right triangle at the knee between the hip and foot, and the repetition would only be counted if a 90 degree angle was achieved. In the novel case, a repetition is judged by determining if a certain angle is met between 3 landmarks on the body. With this simple algorithm, we can use the MediaPipe pose module to identify the target landmarks while using basic trigonometry to determine the angles. The computer can then make the determination of counting a repetition or calling it a no-rep.  
- Creating an algorithm to grade simple movements, such as a squat or a curl, was a trivial challenge. Unsurprisingly, there are many exercises comprised of compound movements where the algorithm falls short. Take for example the clean and jerk exercise. A person will hoist a bar to shoulder level before pushing the weight above their head and locking out their arms for a completed attempt. In this case, we'll use the hand, elbow and shoulder as our 3 landmarks. The arms start at approximately -180 degree, move to 40-60 degrees at the shoulder, and then end around ~160 degrees at lockout. This is complex to judge because the algorithm needs to take into account each movement separately from one another and when to reset.
- Generalizing the algorithm to work with different camera angles and distances is also complex. In theoretical application, the camera would sit on a plane parallel to the athlete either directly in front, back, or to the side. When looking for test data though, most athletes are positioned at an angle to the camera. This caused a distortion in the angle measurement because our model cannot account for this angle and normalize the measured triangle to a parallel plane. I spent much time trying to determine a way around this issue, but came to the conclusion that more data would be required. We would need to know the distance of the athlete from the camera and a measurement of the distance between a fixed landmarks such as the shoulders. With this data, we could then determine the angle of the camera and rotate the landmarks around the z-axis to essentially bring them parallel to the viewing plane. As a workaround to this issue, I opted to just add a buffer to the measured angles to account for the distortion knowing that this would not generalize well to the multitude of camera angles/distances. 
  
  
##### Potential Use Cases
- Software to augment physical therapists. As with many clinicians, physical therapists will be in short supply if patient workloads keep increasing. A system similar to this could them quantify and track patient progress to improve outcomes.  
- Creating smart mirrors at fitness centers. Some people love to look at themselves in the mirror while working out, so why not add an additional layer of useful data metrics.
- Replacing referees at sporting events such as basketball, baseball, or soccer at the local and professional levels. 