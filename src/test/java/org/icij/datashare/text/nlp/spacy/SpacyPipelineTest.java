package org.icij.datashare.text.nlp.spacy;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.MockitoAnnotations.openMocks;

import java.io.Serializable;
import java.util.List;
import org.icij.datashare.PropertiesProvider;
import org.icij.datashare.asynctasks.Task;
import org.icij.datashare.asynctasks.TaskManager;
import org.icij.datashare.text.Language;
import org.icij.datashare.text.NamedEntity;
import org.icij.datashare.text.nlp.NlpTag;
import org.junit.After;
import org.junit.Before;
import org.mockito.Mock;

public class SpacyPipelineTest {
    private AutoCloseable mocks;
    private SpacyPipeline pipeline;

    @Mock
    private TaskManager taskManager;

    @Before
    public void setUp() {
        mocks = openMocks(this);
        pipeline = new SpacyPipeline(new PropertiesProvider(), taskManager);
    }

    @After
    public void tearDown() throws Exception {
        mocks.close();
    }

    void test_should_process() {
        // Given
        Language language = Language.ENGLISH;
        Task<List<List<NlpTag>>> doneTask = new Task<>("taskId", "spacy-ner", null, null);
        doneTask.setResult((Serializable) List.of(List.of(),
            List.of(new SpacyPipeline.SpacyNamedEntity(11, 16, NamedEntity.Category.LOCATION)))
        );
        doReturn(doneTask).when(taskManager).getTask(anyString());
        List<String> batch = List.of("some text", "written in Paris");
        // When
        List<List<NlpTag>> tags = pipeline.processText(batch.stream(), language);
        // Then
        List<List<NlpTag>> expectedTags = List.of(List.of(), List.of(new NlpTag(11, 12, NamedEntity.Category.LOCATION)));
        assertThat(tags).isEqualTo(expectedTags);
    }
}