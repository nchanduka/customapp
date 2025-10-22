**NOTES**

1. Clone the Gir Repo.

2. Make required changes to **generate()** function in the app.py to fit the requirements of the video to be generated.

3. Run the docker commands to build your image:
   
    ```
    docker build -t video-streaming:latest .
  
    docker tag video-streaming nchanduka/video-streaming:latest
    
    docker push nchanduka/video-streaming:latest

5. Now just go ahead make changes to the values.yaml in the helm-chart to include your image name.

   ```
   image:
      repository: <your-docker-repo>/video-streaming
      pullPolicy: Always
      tag: latest

7. Buidl the helm package.

   ```
   helm package .

9. Use the helm-chart to dpeloy the applciation.
