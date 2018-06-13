angular.module('gveApp', [])
    .filter("trust", ['$sce', function($sce) {
        return function(htmlCode){
            return $sce.trustAsHtml(htmlCode);
        }
    }])
    .controller('MainCtrl', ['$scope', '$http', function ($scope, $http) {
        $scope.loading_fact_explainer = false;
        $scope.loading_counterfact_explainer = true;
        $scope.server = "http://localhost:5000";

        $scope.displayExplanation = function (image) {
            return "This is a <span class='class_label'>" + image.class_label.split("_").join(" ") + "</span> because " + image.explanation;
        };

        $scope.explain = function (image_index) {
            $scope.current = image_index;
            $scope.sample_explanation = "Loading...";
            $http.get($scope.server + '/explain/' + $scope.sample_images[$scope.current].id).then(function (response) {
                $scope.sample_explanation = $scope.displayExplanation(response.data);
            });
        };

        $scope.load_sample_images = function () {
            $scope.sample_images = [];
            $http.get($scope.server + '/sample_images/10').then(function (response) {
                $scope.sample_images = response.data;
                $scope.explain(0);
            });
        };

        $scope.load_classes = function () {
            $scope.classes = [];
            $http.get($scope.server + '/classes').then(function (response) {
                $scope.classes = response.data;
                $scope.retry();
            });
        };

        $scope.retry = function () {
            $scope.loading_counterfact_explainer = true;
            $scope.explanation = "";
            $scope.images = [];
            $scope.correct_class_label = "";
            $scope.result = false;
            var class1 = $scope.classes[Math.floor(Math.random() * $scope.classes.length)];
            var index = $scope.classes.indexOf(class1);
            var classes = angular.copy($scope.classes);
            classes.splice(index, 1);
            var class2 = classes[Math.floor(Math.random() * classes.length)];
            $http.get($scope.server + '/counter_factual' + '/' + class1 + '/' + class2).then(function (response) {
                $scope.explanation = response.data.images[0].explanation;
                $scope.correct_class_label = response.data.images[0].class_label;
                $scope.images = shuffle(response.data.images);
                $scope.loading_counterfact_explainer = false;
                $scope.answered = false;
                $scope.result = false;
            });
        };

        $scope.load_sample_images();
        $scope.load_classes();

        function shuffle(array) {
            var currentIndex = array.length, temporaryValue, randomIndex;

            // While there remain elements to shuffle...
            while (0 !== currentIndex) {

                // Pick a remaining element...
                randomIndex = Math.floor(Math.random() * currentIndex);
                currentIndex -= 1;

                // And swap it with the current element.
                temporaryValue = array[currentIndex];
                array[currentIndex] = array[randomIndex];
                array[randomIndex] = temporaryValue;
            }

            return array;
        }

        $scope.select = function (image) {
            $scope.answered = true;
            $scope.result = $scope.correct_class_label == image.class_label;
        };
    }]);
